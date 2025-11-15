# medichain-fl/backend/fl_client/client.py
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from pathlib import Path
import sys
from PIL import Image  # Required for Hugging Face processor

# Add the parent directory (backend) to sys.path to import model.py
sys.path.append(str(Path(__file__).parent.parent))
from model import load_model, load_image_processor  # Import both functions


class HuggingFaceImageFolder(Dataset):
    """
    A wrapper for ImageFolder to apply Hugging Face's image processor.
    """
    def __init__(self, root: str, image_processor):
        self.image_processor = image_processor
        # ImageFolder will load PIL images by default
        self.dataset = datasets.ImageFolder(root)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        processed_inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = processed_inputs['pixel_values'].squeeze(0)
        return pixel_values, label


class PneumoniaClient(fl.client.NumPyClient):
    def __init__(self, hospital_id: str, data_path: str, model_name: str = "dima806/chest_xray_pneumonia_detection", device: str = 'cpu'):
        self.hospital_id = hospital_id
        self.device = device
        
        # Load Hugging Face model
        self.model = load_model(model_name=model_name, freeze_encoder=True).to(device)
        
        # Load processor
        self.image_processor = load_image_processor(model_name=model_name)
        
        # Dataset
        self.trainset = HuggingFaceImageFolder(data_path, image_processor=self.image_processor)
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=8, # Kept the reduced batch size
            shuffle=True,
            num_workers=0, # IMPORTANT CHANGE: Set num_workers to 0 for stability on macOS/Docker
            pin_memory=False # IMPORTANT CHANGE: Set pin_memory to False when not using a GPU
        )
        
        # Loss + optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=0.001
        )

        print(f"[{self.hospital_id}] Initialized with {len(self.trainset)} samples. Device: {self.device}")
        print(f"[{self.hospital_id}] Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def get_parameters(self, config):
        """
        Returns only the model parameters that are trainable (unfrozen).
        """
        print(f"[{self.hospital_id}] Getting trainable parameters.")
        # Find all parameters that have requires_grad = True
        trainable_params = [val.cpu().detach().numpy() for val in self.model.parameters() if val.requires_grad]
        return trainable_params
    
    def set_parameters(self, parameters):
        """
        Sets the model parameters.
        NOTE: This client now receives the FULL model from the server,
        so this function can handle loading the full state dict correctly.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        # IMPORTANT: The client now receives the FULL updated model from the server
        self.set_parameters(parameters)
        self.model.train()
        
        epoch_loss, correct, total = 0.0, 0, 0
        
        for batch_idx, (pixel_values, labels) in enumerate(self.trainloader):
            pixel_values, labels = pixel_values.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits
            
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0 and batch_idx > 0:
                print(f"[{self.hospital_id}] Batch {batch_idx}/{len(self.trainloader)} - Loss: {loss.item():.4f}")
        
        accuracy = 100 * correct / total
        avg_loss = epoch_loss / len(self.trainloader)
        
        print(f"[{self.hospital_id}] Round finished - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
        
        # IMPORTANT: Return ONLY the updated (trainable) parameters
        return self.get_parameters(config={}), len(self.trainloader.dataset), {
            "loss": float(avg_loss),
            "accuracy": float(accuracy)
        }
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for pixel_values, labels in self.trainloader:
                pixel_values, labels = pixel_values.to(self.device), labels.to(self.device)
                outputs = self.model(pixel_values=pixel_values)
                logits = outputs.logits
                loss += self.criterion(logits, labels).item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = loss / len(self.trainloader)
        
        print(f"[{self.hospital_id}] Evaluation - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
        
        return float(avg_loss), total, {"accuracy": float(accuracy)}

def start_client(hospital_id: str, server_address: str = "localhost:8080"):
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    client = PneumoniaClient(
        hospital_id,
        f"/app/data/{hospital_id}",
        device=device
    )

    fl.client.start_client(server_address="flower_server:8080", client=client.to_client())

if __name__ == "__main__":
    import sys
    hospital_id = sys.argv[1] if len(sys.argv) > 1 else "hospital_1"
    start_client(hospital_id)
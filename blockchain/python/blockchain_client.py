from web3 import Web3
import json

RPC = "http://127.0.0.1:8545"
PROXY_ADDRESS = "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0"
ACCOUNT = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

w3 = Web3(Web3.HTTPProvider(RPC))

with open("abi.json") as f:
    abi_data = json.load(f)
    ABI = abi_data["abi"] if isinstance(abi_data, dict) else abi_data

contract = w3.eth.contract(address=PROXY_ADDRESS, abi=ABI)

def log_gradient_update(hash_value):
    nonce = w3.eth.get_transaction_count(ACCOUNT)
    tx = contract.functions.logUpdate(hash_value).build_transaction({
        "from": ACCOUNT,
        "nonce": nonce,
        "gas": 500000,
        "gasPrice": w3.to_wei("1", "gwei"),
    })

    signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt

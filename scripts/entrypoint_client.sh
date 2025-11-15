#!/bin/bash
echo "Starting Flower Client for Hospital: $HOSPITAL_ID"

# Add a delay to allow the server to start up fully before connecting
echo "Waiting 10 seconds for server to initialize..."
sleep 10

exec python backend/fl_client/client.py "$HOSPITAL_ID"
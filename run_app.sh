#!/bin/bash

# Run app.py on port 8501
streamlit run app.py --server.port 8501 &

# Run op.py on port 8502
streamlit run op.py --server.port 8502 &


streamlit run cite.py --server.port 8503 &
# Wait for both processes to complete
wait


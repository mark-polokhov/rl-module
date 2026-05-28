# INSIDE PROTO FOLDER #
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. racing.proto
import torch


def main():
    print("MyLLM project")
    print("========================")

    if torch.cuda.is_available():
        device = torch.device("cuda")

        print("Device:", torch.cuda.get_device_name(device))
        print("CUDA:", torch.version.cuda)
        print("PyTorch:", torch.__version__)

        print("\nEnvironment initialized!")
    else:
        print("CUDA is unavailable!")


if __name__ == "__main__":
    main()
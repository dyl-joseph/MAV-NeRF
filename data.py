from test_replay_dataset import TestReplayDataset

# Create an instance of TestReplayDataset
dataset = TestReplayDataset()

# Optionally, check the length of the dataset
# print("Dataset length:", len(dataset))

# Access and print the first few items
for i in range(min(0,5)):
    item = dataset[i]
    print(f"Item {i}:", item)
from replay_dataset import ReplayDataset

SQL_ROOT = r"C:\Users\dylan\replay_dataset\tests\data\replay_dataset_100.sqlite"
NO_BLOBS_KWARGS = {
    "dataset_root": "",
    "load_images": False,
    "load_depths": False,
    "load_masks": False,
    "load_depth_masks": False,
    "box_crop": False,
}



dataset = ReplayDataset(
    sqlite_metadata_file=SQL_ROOT,
    remove_empty_masks=False,
    frame_data_builder_ReplayFrameDataBuilder_args=NO_BLOBS_KWARGS,
)

print(type(dataset))
print(len(dataset))
for i in range(len(dataset)):
    item = dataset[i]
    print(item.sequence_name)

sequence = 'cat0_seq0'
frame_number = 0
item = dataset[sequence, frame_number]
print(item)
    
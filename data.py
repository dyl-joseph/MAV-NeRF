import matplotlib.pyplot as plt
import numpy as np

from replay_dataset import ReplayDataset

def visualize_dataset_item(dataset, sequence_name, frame_idx):
    """
    Visualize a dataset item at the specified sequence and frame number.
    """
    print(f"Visualizing dataset item for sequence: {sequence_name}, frame: {frame_idx}")
    print("=" * 60)
    
    # Get the item
    item = dataset[sequence_name, frame_idx]
    
    # Print basic information
    print(f"Item type: {type(item)}")
    print(f"Sequence name: {item.sequence_name}")
    print(f"Frame number: {item.frame_number}")
    print(f"Frame timestamp: {item.frame_timestamp}")
    print(f"Sequence category: {item.sequence_category}")
    print(f"Sensor name: {item.sensor_name}")
    print(f"Camera quality score: {item.camera_quality_score}")
    print(f"Point cloud quality score: {item.point_cloud_quality_score}")
    
    # Print image information
    print(f"\nImage information:")
    print(f"  Image size (HxW): {item.image_size_hw}")
    print(f"  Effective image size (HxW): {item.effective_image_size_hw}")
    print(f"  Image path: {item.image_path}")
    print(f"  Has RGB image: {item.image_rgb is not None}")
    
    # Print camera information
    print(f"\nCamera information:")
    print(f"  Has camera: {item.camera is not None}")
    if item.camera is not None:
        print(f"  Camera type: {type(item.camera)}")
        print(f"  Camera device: {item.camera.device}")
        print(f"  Camera R shape: {item.camera.R.shape if hasattr(item.camera, 'R') else 'N/A'}")
        print(f"  Camera T shape: {item.camera.T.shape if hasattr(item.camera, 'T') else 'N/A'}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Dataset Item Visualization\nSequence: {sequence_name}, Frame: {frame_idx}', fontsize=16)
    
    # Plot 1: RGB Image (if available)
    ax1 = axes[0, 0]
    if item.image_rgb is not None:
        # Convert tensor to numpy and transpose if needed
        img = item.image_rgb.detach().cpu().numpy()
        if img.shape[0] == 3:  # CHW format
            img = np.transpose(img, (1, 2, 0))
        # Normalize to 0-1 range if needed
        if img.max() > 1.0:
            img = img / 255.0
        ax1.imshow(img)
        ax1.set_title('RGB Image')
        ax1.axis('off')
    else:
        ax1.text(0.5, 0.5, 'No RGB image available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('RGB Image (Not Available)')
        ax1.axis('off')
    
    # Plot 2: Foreground Mask (if available)
    ax2 = axes[0, 1]
    if item.fg_probability is not None:
        mask = item.fg_probability.detach().cpu().numpy()
        if len(mask.shape) == 3 and mask.shape[0] == 1:
            mask = mask[0]  # Remove channel dimension
        ax2.imshow(mask, cmap='gray')
        ax2.set_title('Foreground Probability Mask')
        ax2.axis('off')
    else:
        ax2.text(0.5, 0.5, 'No mask available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Foreground Mask (Not Available)')
        ax2.axis('off')
    
    # Plot 3: Depth Map (if available)
    ax3 = axes[1, 0]
    if item.depth_map is not None:
        depth = item.depth_map.detach().cpu().numpy()
        if len(depth.shape) == 3 and depth.shape[0] == 1:
            depth = depth[0]  # Remove channel dimension
        im = ax3.imshow(depth, cmap='viridis')
        ax3.set_title('Depth Map')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    else:
        ax3.text(0.5, 0.5, 'No depth map available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Depth Map (Not Available)')
        ax3.axis('off')
    
    # Plot 4: Point Cloud (if available)
    ax4 = axes[1, 1]
    if item.sequence_point_cloud is not None:
        pcl = item.sequence_point_cloud.detach().cpu().numpy()
        # Sample points if too many
        if len(pcl) > 10000:
            indices = np.random.choice(len(pcl), 10000, replace=False)
            pcl = pcl[indices]
        
        # 3D scatter plot
        ax4.scatter(pcl[:, 0], pcl[:, 1], c=pcl[:, 2], s=0.1, alpha=0.6)
        ax4.set_title('Point Cloud (Top View)')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_aspect('equal')
    else:
        ax4.text(0.5, 0.5, 'No point cloud available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Point Cloud (Not Available)')
        ax4.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print additional details
    print(f"\nDetailed tensor shapes:")
    for attr_name in dir(item):
        if not attr_name.startswith('_'):
            attr = getattr(item, attr_name)
            if hasattr(attr, 'shape') and attr is not None:
                print(f"  {attr_name}: {attr.shape} ({type(attr)})")
            elif hasattr(attr, '__len__') and not isinstance(attr, str) and attr is not None:
                print(f"  {attr_name}: {len(attr)} ({type(attr)})")

SQL_ROOT = r"C:\Users\dylan\replay_dataset\tests\data\replay_dataset_100.sqlite"
SET_LIST_FILE = r"C:\Users\dylan\replay_dataset\tests\data\set_lists_100.json"
NO_BLOBS_KWARGS = {
    "dataset_root": r"C:\Users\dylan",
    "load_images": False,
    "load_depths": False,
    "load_masks": False,
    "load_depth_masks": False,
    "box_crop": False,
}
dataset = ReplayDataset(
            sqlite_metadata_file=SQL_ROOT,
            remove_empty_masks=False,
            subset_lists_file=SET_LIST_FILE,
            limit_to=100,  # force sorting
            subsets=["train", "test"],
            frame_data_builder_ReplayFrameDataBuilder_args=NO_BLOBS_KWARGS,
        )
dataset = ReplayDataset(
            sqlite_metadata_file=SQL_ROOT,
            remove_empty_masks=False,
            frame_data_builder_ReplayFrameDataBuilder_args=NO_BLOBS_KWARGS,
)
sequence = 'cat1_seq2'
frame_number = 0
visualize_dataset_item(dataset, sequence, frame_number) 
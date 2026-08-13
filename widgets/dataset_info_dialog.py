from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from panopticon.data_engine.coco_dataset_stats import (
    compute_split_stats,
    summarize_distribution,
)


class DatasetInfoDialog(QDialog):
    def __init__(self, parent=None, coco_data=None, tr_func=None):
        super().__init__(parent)
        self.tr = tr_func if tr_func else lambda text: text
        self.setWindowTitle(self.tr("dataset_info_dialog_title"))
        self.coco_data = coco_data if coco_data is not None else {}

        # Compute statistics
        try:
            self.stats = compute_split_stats("dataset", self.coco_data, 32.0, 96.0)
            self.categories_meta = {cat["id"]: cat for cat in self.coco_data.get("categories", [])}
        except Exception as e:
            self.stats = None
            self.categories_meta = {}
            print(f"Error computing dataset statistics: {e}")

        self.init_ui()

    def init_ui(self):
        # Create scroll area for large content
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)

        # Display info field
        info = self.coco_data.get("info", {})
        info_label = QLabel(self.tr("dataset_info_label"))
        info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(info_label)

        if info:
            for key, value in info.items():
                layout.addWidget(QLabel(f"  <b>{key}:</b> {value}"))
        else:
            layout.addWidget(QLabel(self.tr("no_info_available")))

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        # Basic Statistics
        if self.stats:
            self.add_basic_stats(layout)
            self.add_annotation_stats(layout)
            self.add_bbox_stats(layout)
            self.add_image_resolution_stats(layout)
            self.add_category_stats(layout)
            self.add_cooccurrence_stats(layout)

        # Add OK button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)

        self.setMinimumWidth(600)
        self.setMinimumHeight(800)
        self.resize(800, 1000)

    def add_basic_stats(self, layout):
        """Add basic dataset statistics section."""
        if not self.stats:
            return

        title = QLabel("Basic Statistics")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
        layout.addWidget(title)

        stats = self.stats
        layout.addWidget(QLabel(f"Images: {stats.images:,}"))
        layout.addWidget(QLabel(f"Annotations: {stats.annotations:,}"))
        layout.addWidget(QLabel(f"Categories: {stats.categories}"))
        pct_no_ann = (stats.images_with_no_annotations / stats.images * 100) if stats.images else 0.0
        layout.addWidget(QLabel(f"Images with no annotations: {stats.images_with_no_annotations} ({pct_no_ann:.1f}%)"))
        pct_with_ann = (stats.images_with_annotations / stats.images * 100) if stats.images else 0.0
        layout.addWidget(QLabel(f"Images with annotations: {stats.images_with_annotations} ({pct_with_ann:.1f}%)"))

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

    def add_annotation_stats(self, layout):
        """Add annotation statistics section."""
        if not self.stats:
            return

        title = QLabel("Annotation Statistics")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
        layout.addWidget(title)

        stats = self.stats
        layout.addWidget(QLabel(f"Average annotations per image: {stats.avg_annotations_per_image:.2f}"))
        layout.addWidget(QLabel(f"Median annotations per image: {stats.median_annotations_per_image:.2f}"))
        layout.addWidget(QLabel(f"Maximum annotations in image: {stats.max_annotations_in_image}"))
        layout.addWidget(QLabel(f"Annotations with bbox outside image: {stats.bbox_outside_image}"))
        layout.addWidget(QLabel(f"Estimated duplicate boxes: {stats.duplicate_boxes_estimated}"))

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

    def add_bbox_stats(self, layout):
        """Add bounding box statistics section."""
        if not self.stats:
            return

        title = QLabel("Bounding Box Statistics")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
        layout.addWidget(title)

        stats = self.stats
        bbox_dist = summarize_distribution(stats.bbox_areas)

        layout.addWidget(QLabel("Area Distribution (pixels²):"))
        layout.addWidget(QLabel(f"  Count: {bbox_dist['count']:.0f}"))
        layout.addWidget(QLabel(f"  Mean: {bbox_dist['mean']:.2f}"))
        layout.addWidget(QLabel(f"  Median: {bbox_dist['median']:.2f}"))
        layout.addWidget(QLabel(f"  Std Dev: {bbox_dist['std']:.2f}"))
        layout.addWidget(QLabel(f"  Min: {bbox_dist['min']:.2f}"))
        layout.addWidget(QLabel(f"  Max: {bbox_dist['max']:.2f}"))
        layout.addWidget(QLabel(f"  10th percentile: {bbox_dist['p10']:.2f}"))
        layout.addWidget(QLabel(f"  90th percentile: {bbox_dist['p90']:.2f}"))

        ar_dist = summarize_distribution(stats.aspect_ratios)
        layout.addWidget(QLabel("Aspect Ratio Distribution (w/h):"))
        layout.addWidget(QLabel(f"  Count: {ar_dist['count']:.0f}"))
        layout.addWidget(QLabel(f"  Mean: {ar_dist['mean']:.4f}"))
        layout.addWidget(QLabel(f"  Median: {ar_dist['median']:.4f}"))
        layout.addWidget(QLabel(f"  Std Dev: {ar_dist['std']:.4f}"))
        layout.addWidget(QLabel(f"  Min: {ar_dist['min']:.4f}"))
        layout.addWidget(QLabel(f"  Max: {ar_dist['max']:.4f}"))

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

    def add_image_resolution_stats(self, layout):
        """Add image resolution statistics section."""
        if not self.stats:
            return

        title = QLabel("Image Resolution Statistics")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
        layout.addWidget(title)

        stats = self.stats
        res_stats = stats.image_resolution_stats

        layout.addWidget(QLabel(f"Images with resolution data: {res_stats['count']:.0f}"))
        layout.addWidget(QLabel(f"Width - Mean: {res_stats['width_mean']:.1f}, Median: {res_stats['width_median']:.1f}"))
        layout.addWidget(QLabel(f"Height - Mean: {res_stats['height_mean']:.1f}, Median: {res_stats['height_median']:.1f}"))
        layout.addWidget(QLabel(f"Width range: {res_stats['width_min']:.0f} - {res_stats['width_max']:.0f}"))
        layout.addWidget(QLabel(f"Height range: {res_stats['height_min']:.0f} - {res_stats['height_max']:.0f}"))
        layout.addWidget(QLabel(f"Average megapixels: {res_stats['megapixels_mean']:.3f}"))

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

    def add_category_stats(self, layout):
        """Add category statistics section."""
        if not self.stats:
            return

        title = QLabel("Category Statistics (All Categories)")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
        layout.addWidget(title)

        stats = self.stats
        # Sort categories by annotation count (show all)
        sorted_cats = sorted(stats.category_annotation_counts.items(), key=lambda x: x[1], reverse=True)

        for cat_id, ann_count in sorted_cats:
            cat_name = self.categories_meta.get(cat_id, {}).get("name", f"Category {cat_id}")
            img_count = stats.category_image_counts.get(cat_id, 0)
            area_stats = stats.per_category_area_stats.get(cat_id, {})
            mean_area = area_stats.get("mean", 0)

            layout.addWidget(QLabel(f"  {cat_name}: {ann_count:,} annotations, {img_count:,} images, mean area: {mean_area:.1f}"))

        # Add imbalance metrics
        counts = list(stats.category_annotation_counts.values())
        if counts:
            # Simple Gini coefficient calculation
            counts_sorted = sorted(counts)
            n = len(counts_sorted)
            cumsum = sum((i + 1) * val for i, val in enumerate(counts_sorted))
            gini = (2 * cumsum) / (n * sum(counts_sorted)) - (n + 1) / n if sum(counts_sorted) > 0 else 0

            top_count = max(counts)
            median_count = sorted(counts)[len(counts) // 2]
            ratio = top_count / median_count if median_count > 0 else float("inf")

            layout.addWidget(QLabel(f"Category imbalance - Gini coefficient: {gini:.4f}"))
            layout.addWidget(QLabel(f"Top category / median category ratio: {ratio:.2f}"))

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

    def add_cooccurrence_stats(self, layout):
        """Add category co-occurrence statistics section."""
        if not self.stats:
            return

        title = QLabel("Category Co-occurrence (Top 15 pairs)")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
        layout.addWidget(title)

        stats = self.stats
        # Sort co-occurrence pairs by count
        sorted_pairs = sorted(stats.cooccurrence.items(), key=lambda x: x[1], reverse=True)[:15]

        for (cat1_id, cat2_id), count in sorted_pairs:
            cat1_name = self.categories_meta.get(cat1_id, {}).get("name", f"Category {cat1_id}")
            cat2_name = self.categories_meta.get(cat2_id, {}).get("name", f"Category {cat2_id}")

            if cat1_id == cat2_id:
                pair_name = f"{cat1_name} (self-cooccurrence)"
            else:
                pair_name = f"{cat1_name} + {cat2_name}"

            layout.addWidget(QLabel(f"  {pair_name}: {count:,}"))

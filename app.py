"""
Vehicle AI - Interactive Streamlit Dashboard
Real-time Vehicle Detection, Tracking, Bidirectional Counting & Traffic Analytics
"""

from collections import Counter, defaultdict, deque
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="Vehicle AI Analytics Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00D4FF;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #A0AEC0;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #1A202C;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #2D3748;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #00D4FF;
    }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
WEIGHTS_CLEAN = ROOT / "weights" / "vehicle_yolov8s_best.pt"
WEIGHTS_ENHANCED = ROOT / "runs" / "vehicle" / "roboflow_v2_enhanced" / "weights" / "best.pt"
WEIGHTS_BASELINE = ROOT / "runs" / "vehicle" / "roboflow_v2" / "weights" / "best.pt"
DEFAULT_WEIGHTS = WEIGHTS_CLEAN if WEIGHTS_CLEAN.exists() else (WEIGHTS_ENHANCED if WEIGHTS_ENHANCED.exists() else WEIGHTS_BASELINE)
SAMPLE_VIDEO = ROOT / "traffic.mp4"
TRACKER_CFG = ROOT / "configs" / "tracker_traffic.yaml" if (ROOT / "configs" / "tracker_traffic.yaml").exists() else ROOT / "tracker_traffic.yaml"

CLASS_NAMES = ("bus", "car", "motorbike", "truck")
CLASS_COLORS = {
    "bus": (255, 128, 0),        # BGR: Orange-blue
    "car": (0, 220, 255),        # Cyan-yellow
    "motorbike": (255, 0, 200),  # Magenta
    "truck": (0, 255, 128),      # Mint Green
}


@st.cache_resource
def load_yolo_model(weights_path: str):
    """Cache the loaded YOLO model to avoid reloading on every rerun."""
    return YOLO(weights_path)


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/traffic-jam.png", width=72)
    st.title("Settings & Controls")

    device_str = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "System CPU"
    st.caption(f"⚡ Device: **{device_name}** ({device_str})")

    st.divider()

    model_option = st.selectbox(
        "🧠 Select Model Engine",
        [
            "🌟 General Vehicle AI (COCO Pretrained - Recommended)",
            "🎯 Custom Fine-Tuned (Roboflow v2 Enhanced)"
        ],
        index=0,
        help="General Vehicle AI works flawlessly on any highway/street. Custom model is trained on the Roboflow dataset."
    )

    mode = st.radio(
        "Select Operation Mode",
        ["🎥 Video Tracking & Counting", "📸 Image Vehicle Detection"],
        index=0
    )

    st.divider()
    st.subheader("Model Parameters")

    conf_thresh = st.slider(
        "Detection Confidence",
        min_value=0.10, max_value=0.90, value=0.25, step=0.05,
        help="Higher values reduce false positives; lower values detect fainter/smaller vehicles."
    )

    if mode == "🎥 Video Tracking & Counting":
        line_ratio = st.slider(
            "Counting Line Position",
            min_value=0.20, max_value=0.85, value=0.60, step=0.02,
            help="Vertical position of the horizontal counting line as fraction of video height."
        )
        band_ratio = st.slider(
            "Hysteresis Deadband",
            min_value=0.01, max_value=0.05, value=0.02, step=0.005,
            help="Buffer zone around the counting line to prevent boundary jitter."
        )
        imgsz = st.selectbox(
            "Inference Resolution",
            [640, 800, 960, 1080],
            index=2,
            help="Higher resolution drastically improves distant/small vehicle detection."
        )
        show_trails = st.checkbox("Show Trajectory Trails", value=True)
    else:
        imgsz = st.selectbox("Inference Resolution", [640, 800, 960, 1080], index=2)

    st.divider()
    st.caption("AI Vehicle Vision • YOLOv8 + ByteTrack")


# ==========================================
# MAIN INTERFACE
# ==========================================
st.markdown('<div class="main-header">🚗 Vehicle AI: Intelligent Traffic Analytics</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-class vehicle detection, persistent ByteTrack tracking, and bidirectional line-crossing counting</div>',
    unsafe_allow_html=True
)

use_coco = "COCO" in model_option
model_path = "yolov8s.pt" if use_coco else str(DEFAULT_WEIGHTS)
filter_classes = [2, 3, 5, 7] if use_coco else None  # car, motorcycle, bus, truck in COCO
model = load_yolo_model(model_path)


# -------------------------------------------------------------
# MODE 1: IMAGE DETECTION
# -------------------------------------------------------------
if mode == "📸 Image Vehicle Detection":
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"])

    col_btn, _ = st.columns([1, 4])
    use_sample = col_btn.button("Use Sample Traffic Image")

    input_img = None
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif use_sample:
        sample_img_path = ROOT / "traffic_detected.jpg"
        if sample_img_path.exists():
            input_img = cv2.imread(str(sample_img_path))
        else:
            st.warning("Sample image not found, please upload an image.")

    if input_img is not None:
        with st.spinner("Running vehicle detection..."):
            rgb_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            predict_kwargs = {"source": rgb_img, "conf": conf_thresh, "imgsz": imgsz, "verbose": False}
            if filter_classes:
                predict_kwargs["classes"] = filter_classes
            results = model.predict(**predict_kwargs)
            result = results[0]

            annotated_rgb = result.plot()
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            # Class count breakdown
            class_counts = Counter()
            if result.boxes and len(result.boxes) > 0:
                cids = result.boxes.cls.int().cpu().tolist()
                for cid in cids:
                    class_counts[result.names[cid]] += 1

        total_detected = sum(class_counts.values())

        # Metrics display
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Vehicles", total_detected)
        m2.metric("Cars", class_counts["car"])
        m3.metric("Buses", class_counts["bus"])
        m4.metric("Trucks", class_counts["truck"])
        m5.metric("Motorbikes", class_counts["motorbike"])

        st.divider()

        # Side-by-side image comparison
        c_orig, c_pred = st.columns(2)
        with c_orig:
            st.caption("Original Image")
            st.image(rgb_img, use_container_width=True)
        with c_pred:
            st.caption("Annotated Predictions")
            st.image(annotated_rgb, use_container_width=True)

        # Download button
        _, buffer = cv2.imencode(".jpg", annotated_bgr)
        st.download_button(
            label="📥 Download Annotated Image",
            data=buffer.tobytes(),
            file_name="vehicle_detection_result.jpg",
            mime="image/jpeg"
        )


# -------------------------------------------------------------
# MODE 2: VIDEO TRACKING & BIDIRECTIONAL COUNTING
# -------------------------------------------------------------
else:
    st.subheader("Upload Traffic Video or Run Demo")
    video_file = st.file_uploader("Upload an MP4, MOV, or AVI video", type=["mp4", "mov", "avi"])

    c_sample, _ = st.columns([2, 5])
    use_sample_video = c_sample.checkbox("Use Bundled 'traffic.mp4' Demo Video", value=(video_file is None))

    video_source_path = None
    if video_file is not None and not use_sample_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        video_source_path = tfile.name
    elif use_sample_video:
        if SAMPLE_VIDEO.exists():
            video_source_path = str(SAMPLE_VIDEO)
        else:
            st.error("Bundled traffic.mp4 not found. Please upload a video.")

    if video_source_path:
        st.info(f"Loaded Video Source: **{Path(video_source_path).name}**")

        start_btn = st.button("🚀 Start Vehicle Tracking & Counting", type="primary")

        if start_btn:
            cap = cv2.VideoCapture(video_source_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

            line_y = int(height * line_ratio)
            band_px = max(2, int(height * band_ratio))
            upper_y = max(0, line_y - band_px)
            lower_y = min(height - 1, line_y + band_px)

            # Temporary output video file
            out_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            writer = cv2.VideoWriter(
                out_temp.name,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (width, height)
            )

            # Live UI Placeholders
            progress_bar = st.progress(0, text="Initializing tracker...")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
            kpi_inbound = kpi_col1.empty()
            kpi_outbound = kpi_col2.empty()
            kpi_total = kpi_col3.empty()
            kpi_active = kpi_col4.empty()
            kpi_fps = kpi_col5.empty()

            preview_slot = st.empty()

            # State tracking
            trajectories = defaultdict(lambda: deque(maxlen=30))
            class_history = defaultdict(lambda: deque(maxlen=20))
            last_stable_side = {}
            counted_directions = set()
            last_seen_frame = {}

            direction_counts = Counter()
            class_crossing_counts = Counter()
            frame_idx = 0
            t_start = time.time()
            rolling_fps = fps

            tracker_arg = str(TRACKER_CFG) if TRACKER_CFG.exists() else "bytetrack.yaml"
            device = 0 if torch.cuda.is_available() else "cpu"

            try:
                while True:
                    t_frame_start = time.time()
                    ok, frame = cap.read()
                    if not ok:
                        break

                    frame_idx += 1

                    track_kwargs = {
                        "source": frame,
                        "persist": True,
                        "tracker": tracker_arg,
                        "conf": conf_thresh,
                        "imgsz": imgsz,
                        "device": device,
                        "verbose": False
                    }
                    if filter_classes:
                        track_kwargs["classes"] = filter_classes
                    results = model.track(**track_kwargs)
                    result = results[0]

                    # Draw line and deadband
                    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.line(frame, (0, upper_y), (width, upper_y), (255, 180, 0), 1, cv2.LINE_AA)
                    cv2.line(frame, (0, lower_y), (width, lower_y), (255, 180, 0), 1, cv2.LINE_AA)

                    boxes = result.boxes
                    active_count = 0

                    if boxes is not None and boxes.id is not None:
                        coords = boxes.xyxy.cpu().numpy()
                        tids = boxes.id.int().cpu().tolist()
                        cids = boxes.cls.int().cpu().tolist()
                        confs = boxes.conf.cpu().tolist()
                        active_count = len(tids)

                        for box, tid, cid, conf in zip(coords, tids, cids, confs):
                            x1, y1, x2, y2 = map(int, box)
                            contact_x = (x1 + x2) // 2
                            contact_y = y2

                            detected_name = result.names[cid]
                            class_history[tid].append(detected_name)
                            stable_name = Counter(class_history[tid]).most_common(1)[0][0]

                            trajectories[tid].append((contact_x, contact_y))
                            last_seen_frame[tid] = frame_idx

                            # Hysteresis state transition
                            if contact_y < upper_y:
                                current_side = -1
                            elif contact_y > lower_y:
                                current_side = 1
                            else:
                                current_side = None

                            if current_side is not None:
                                prev_side = last_stable_side.get(tid)
                                if prev_side is not None and prev_side != current_side:
                                    direction = "A->B" if (prev_side == -1 and current_side == 1) else "B->A"
                                    count_key = (tid, direction)

                                    if count_key not in counted_directions:
                                        counted_directions.add(count_key)
                                        direction_counts[direction] += 1
                                        class_crossing_counts[stable_name] += 1

                                last_stable_side[tid] = current_side

                            is_counted = any(t == tid for t, _ in counted_directions)
                            box_color = (0, 255, 100) if is_counted else CLASS_COLORS.get(stable_name, (0, 200, 255))

                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                            cv2.circle(frame, (contact_x, contact_y), 4, (0, 0, 255), -1, cv2.LINE_AA)

                            badge = f"{stable_name} #{tid} {conf:.2f}"
                            (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), box_color, -1)
                            cv2.putText(frame, badge, (x1 + 4, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 15, 20), 1, cv2.LINE_AA)

                            if show_trails and len(trajectories[tid]) > 1:
                                pts = np.array(list(trajectories[tid]), dtype=np.int32).reshape((-1, 1, 2))
                                cv2.polylines(frame, [pts], isClosed=False, color=box_color, thickness=2, lineType=cv2.LINE_AA)

                    # Prune stale tracks
                    if frame_idx % 60 == 0:
                        stale = [t for t, f in last_seen_frame.items() if frame_idx - f > 120]
                        for t in stale:
                            trajectories.pop(t, None)
                            class_history.pop(t, None)
                            last_stable_side.pop(t, None)
                            last_seen_frame.pop(t, None)

                    writer.write(frame)

                    # Update rolling FPS
                    f_time = time.time() - t_frame_start
                    if f_time > 0:
                        rolling_fps = 0.9 * rolling_fps + 0.1 * (1.0 / f_time)

                    # Live UI update (every 4 frames for performance)
                    if frame_idx % 4 == 0 or frame_idx == total_video_frames:
                        kpi_inbound.metric("Inbound (A->B)", direction_counts["A->B"])
                        kpi_outbound.metric("Outbound (B->A)", direction_counts["B->A"])
                        kpi_total.metric("Total Crossings", len(counted_directions))
                        kpi_active.metric("Active in View", active_count)
                        kpi_fps.metric("Processing FPS", f"{rolling_fps:.1f}")

                        pct = min(1.0, frame_idx / total_video_frames)
                        progress_bar.progress(pct, text=f"Processing Frame {frame_idx}/{total_video_frames} ({rolling_fps:.1f} FPS)")

                        # Stream sample frame
                        preview_frame_rgb = cv2.cvtColor(cv2.resize(frame, (960, 540)), cv2.COLOR_BGR2RGB)
                        preview_slot.image(preview_frame_rgb, channels="RGB", use_container_width=True)

            finally:
                cap.release()
                writer.release()

            progress_bar.progress(1.0, text="Processing Complete! 🎉")
            st.success("Vehicle tracking and counting finished successfully!")

            st.divider()
            st.subheader("📊 Final Traffic Analytics Summary")

            col_chart, col_df = st.columns([1, 1])

            with col_chart:
                st.caption("Vehicle Distribution by Class")
                df_counts = pd.DataFrame({
                    "Vehicle Type": [c.capitalize() for c in CLASS_NAMES],
                    "Crossings": [class_crossing_counts[c] for c in CLASS_NAMES]
                })
                st.bar_chart(df_counts.set_index("Vehicle Type"))

            with col_df:
                st.caption("Traffic Flow Metrics")
                summary_data = {
                    "Metric": [
                        "Total Inbound (A -> B)",
                        "Total Outbound (B -> A)",
                        "Total Line Crossings",
                        "Total Frames Processed",
                        "Average Inference FPS"
                    ],
                    "Value": [
                        direction_counts["A->B"],
                        direction_counts["B->A"],
                        len(counted_directions),
                        frame_idx,
                        f"{(frame_idx / (time.time() - t_start)):.1f}"
                    ]
                }
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

                # Export CSV button
                csv_bytes = df_counts.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Analytics CSV Report",
                    data=csv_bytes,
                    file_name="traffic_analytics_report.csv",
                    mime="text/csv"
                )

            # Download annotated video button
            with open(out_temp.name, "rb") as f:
                video_bytes = f.read()
            st.download_button(
                label="🎥 Download Full Annotated Video",
                data=video_bytes,
                file_name="annotated_traffic_counted.mp4",
                mime="video/mp4"
            )

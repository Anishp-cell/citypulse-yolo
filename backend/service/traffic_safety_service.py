import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import tempfile
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import shutil
import google.generativeai as genai

# ==================== CONFIG ====================
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

CLASS_NAMES = {
    0: 'no_accident',
    1: 'minor_accident',
    2: 'moderate_accident',
    3: 'severe_accident',
    4: 'totaled_vehicle',
    5: 'pothole'
}

AUTHORITY_CONTACTS = {
    'pothole': {
        'email': 'demo.roads.dept@example.com',
        'phone': '100 (Municipal Helpline)',
        'department': 'Roads & Infrastructure Department'
    },
    'accident': {
        'email': 'demo.emergency@example.com',
        'phone': '108 (Ambulance) / 100 (Police)',
        'department': 'Emergency Services'
    }
}

ACCIDENT_GUIDANCE = {
    'no_accident': {
        'severity': 'None',
        'immediate_actions': [],
        'recommendations': ['No accident detected. Drive safely!'],
        'emergency_level': 0,
        'warning_signs': [],
        'do_not_do': []
    },
    'minor_accident': {
        'severity': 'Minor',
        'immediate_actions': [
            '✓ Ensure safety - Turn on hazard lights',
            '✓ Check for injuries',
            '✓ Move vehicles if safe',
            '✓ Call police for report',
            '✓ Exchange info',
            '✓ Document scene'
        ],
        'recommendations': [
            'Visit doctor within 24-48 hours',
            'Monitor for whiplash symptoms',
            'Report to insurance'
        ],
        'emergency_level': 1,
        'warning_signs': ['Headache', 'Dizziness', 'Nausea'],
        'do_not_do': ['Leave scene', 'Admit fault', 'Sign documents without reading']
    },
    'moderate_accident': {
        'severity': 'Moderate',
        'immediate_actions': [
            '🚨 CALL 108/911',
            '✓ Scene safety',
            '✓ Do not move injured unless necessary',
            '✓ Check consciousness and breathing',
            '✓ Control bleeding'
        ],
        'recommendations': ['Apply pressure to bleeding', 'Keep victim warm', 'Monitor vitals'],
        'emergency_level': 2,
        'warning_signs': ['Loss of consciousness', 'Difficulty breathing', 'Severe pain'],
        'do_not_do': ['Move victim unconditionally', 'Give food/water', 'Remove helmet from motorcyclist']
    },
    'severe_accident': {
        'severity': 'Severe',
        'immediate_actions': [
            '🚨🚨 CALL 108/911 IMMEDIATELY',
            '⚠️ SCENE SAFETY',
            '✓ DO NOT MOVE VICTIMS',
            '✓ Check Airway, Breathing, Circulation',
            '✓ Start CPR if needed',
            '✓ Spinal stabilization'
        ],
        'recommendations': ['CPR Protocol', 'Severe Bleeding Control', 'Treat for Shock'],
        'emergency_level': 3,
        'warning_signs': ['No pulse/breathing', 'Uncontrolled bleeding', 'Unresponsiveness'],
        'do_not_do': ['Move victim', 'Remove impaled objects', 'Give fluids']
    },
    'totaled_vehicle': {
        'severity': 'Critical',
        'immediate_actions': [
            '🚨🚨🚨 CRITICAL EMERGENCY CALL',
            '⚠️ Check for fire/fuel leaks',
            '✓ Rapid Triage',
            '✓ Treat most critical first'
        ],
        'recommendations': ['CPR', 'Tourniquet application', 'Evacuation if fire imminent'],
        'emergency_level': 4,
        'warning_signs': ['Cardiac arrest', 'Fire', 'Multiple casualties'],
        'do_not_do': ['Enter unsafe scene', 'Move trapped victims without tools (unless fire)']
    },
    'pothole': {
        'severity': 'Infrastructure Hazard',
        'immediate_actions': ['Report location', 'Drive slowly', 'Warn other drivers if possible'],
        'recommendations': ['Submit photo to municipal app', 'Check tire/suspension damage'],
        'emergency_level': 0,
        'warning_signs': [],
        'do_not_do': ['Stop in middle of road to inspect']
    }
}

class TrafficSafetyService:
    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.model = None
        self.device = self._get_device()
        self._load_model()
        self._init_gemini()

    def _init_gemini(self):
        """Initialize Gemini API client if API key is available."""
        self.gemini_model = None
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite')
                print("Gemini 2.0 Flash Lite initialized successfully.")
            except Exception as e:
                print(f"Gemini init failed: {e}")
                self.gemini_model = None
        else:
            print("No GEMINI_API_KEY set — AI-enhanced guidance disabled, using structured fallback.")

    def _get_device(self):
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available(): # For Mac
            return 'mps'
        return 'cpu'

    def _load_model(self):
        try:
            print(f"Loading YOLO model from {self.weights_path} on {self.device}...")
            self.model = YOLO(self.weights_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def detect_image(self, image_path: str):
        """
        Process an image from a file path.
        Returns: Dict with processed image path, detections list, and guidance.
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not read image"}

            # Predict
            results = self.model(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device=self.device, verbose=False)[0]
            
            # Process results
            detections = []
            highest_severity_cls = 'no_accident'
            highest_emergency_level = 0
            
            # Draw detections
            output_img = img.copy()
            
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf.item())
                cls_id = int(box.cls.item())
                cls_name = CLASS_NAMES.get(cls_id, 'unknown')
                
                # Filter 'no_accident' from list but keep logic if needed
                if cls_name != 'no_accident':
                    detections.append({'class': cls_name, 'confidence': conf, 'bbox': [x1, y1, x2, y2]})
                    
                    # Update severity
                    severity_info = ACCIDENT_GUIDANCE.get(cls_name, ACCIDENT_GUIDANCE.get('pothole'))
                    current_level = severity_info.get('emergency_level', 0)
                    if current_level > highest_emergency_level:
                        highest_emergency_level = current_level
                        highest_severity_cls = cls_name

                # Color logic
                color = (0, 255, 0)
                if 'minor' in cls_name: color = (0, 255, 255)
                elif 'moderate' in cls_name: color = (0, 165, 255)
                elif 'severe' in cls_name or 'totaled' in cls_name: color = (0, 0, 255)
                elif 'pothole' in cls_name: color = (255, 255, 0)
                
                # Draw
                cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(output_img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(output_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Get Guidance
            guidance = self.get_guidance(highest_severity_cls, detections)
            
            return {
                "success": True,
                "processed_image": output_img,
                "detections": detections,
                "highest_severity": highest_severity_cls,
                "guidance": guidance
            }

        except Exception as e:
            return {"error": str(e)}

    def get_guidance(self, severity_class, detections):
        """Get structured guidance + Gemini AI-enhanced summary."""
        base = ACCIDENT_GUIDANCE.get(severity_class, ACCIDENT_GUIDANCE['no_accident'])
        
        # Deep copy to avoid mutating the template
        guidance = {k: v for k, v in base.items()}

        # Skip AI for no-incident cases
        if severity_class == 'no_accident':
            return guidance

        # Try Gemini AI enhancement
        if self.gemini_model:
            try:
                det_summary = ""
                if isinstance(detections, list) and len(detections) > 0:
                    det_summary = f"{len(detections)} objects detected: " + ", ".join(
                        [f"{d['class'].replace('_', ' ')} ({d['confidence']:.0%})" for d in detections[:5]]
                    )
                else:
                    det_summary = "Single incident detected"

                prompt = (
                    f"You are an emergency medical AI assistant. A road incident has been detected by computer vision.\n\n"
                    f"INCIDENT TYPE: {severity_class.replace('_', ' ').title()}\n"
                    f"SEVERITY: {guidance['severity']}\n"
                    f"DETECTIONS: {det_summary}\n\n"
                    f"Provide a brief, actionable 3-4 sentence summary covering:\n"
                    f"1. The single most critical immediate action to take\n"
                    f"2. Key danger signs to watch for\n"
                    f"3. Whether/when to call emergency services (108/911)\n\n"
                    f"Keep it concise, direct, and focused on saving lives. "
                    f"Use simple language a non-medical bystander can understand."
                )

                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=200,
                    )
                )

                if response and response.text and len(response.text.strip()) > 20:
                    guidance['llm_enhanced'] = response.text.strip()
                    print(f"Gemini guidance generated ({len(guidance['llm_enhanced'])} chars)")

            except Exception as e:
                print(f"Gemini API error: {e}")

        # Fallback if Gemini didn't produce output
        if 'llm_enhanced' not in guidance:
            guidance['llm_enhanced'] = (
                f"💡 This is a {guidance['severity'].lower()}-severity incident. "
                f"{'Call 108/911 immediately. ' if guidance['emergency_level'] >= 2 else ''}"
                f"Focus on the immediate actions listed below. "
                f"{'Watch for warning signs and do not move victims unless absolutely necessary.' if guidance['emergency_level'] >= 2 else 'Document the scene and exchange information.'}"
            )

        return guidance

    def detect_video(self, video_path: str, sample_fps: float = 1.0):
        """
        Process a video file by sampling frames at `sample_fps` rate.
        Returns aggregated detections, worst-severity keyframe, and timeline.
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "Could not open video file"}

            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, int(video_fps / sample_fps))
            duration = total_frames / video_fps if video_fps > 0 else 0

            all_detections = []
            timeline = []
            highest_severity_cls = 'no_accident'
            highest_emergency_level = 0
            worst_frame = None
            worst_frame_detections = []
            frames_processed = 0

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    # Run detection on this frame
                    results = self.model(
                        frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                        device=self.device, verbose=False
                    )[0]

                    frame_detections = []
                    frame_emergency_level = 0
                    frame_severity = 'no_accident'
                    annotated_frame = frame.copy()

                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf.item())
                        cls_id = int(box.cls.item())
                        cls_name = CLASS_NAMES.get(cls_id, 'unknown')

                        if cls_name != 'no_accident':
                            det = {
                                'class': cls_name,
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2],
                                'timestamp': round(frame_idx / video_fps, 2)
                            }
                            frame_detections.append(det)
                            all_detections.append(det)

                            severity_info = ACCIDENT_GUIDANCE.get(cls_name, ACCIDENT_GUIDANCE.get('pothole'))
                            current_level = severity_info.get('emergency_level', 0)
                            if current_level > frame_emergency_level:
                                frame_emergency_level = current_level
                                frame_severity = cls_name

                        # Draw bounding box
                        color = (0, 255, 0)
                        if 'minor' in cls_name: color = (0, 255, 255)
                        elif 'moderate' in cls_name: color = (0, 165, 255)
                        elif 'severe' in cls_name or 'totaled' in cls_name: color = (0, 0, 255)
                        elif 'pothole' in cls_name: color = (255, 255, 0)

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{cls_name} {conf:.2f}"
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    # Record timeline entry if something was found
                    if frame_detections:
                        timeline.append({
                            'timestamp': round(frame_idx / video_fps, 2),
                            'severity': frame_severity,
                            'count': len(frame_detections)
                        })

                    # Track worst frame
                    if frame_emergency_level > highest_emergency_level:
                        highest_emergency_level = frame_emergency_level
                        highest_severity_cls = frame_severity
                        worst_frame = annotated_frame
                        worst_frame_detections = frame_detections

                    frames_processed += 1

                frame_idx += 1

            cap.release()

            # If no incidents found, use last sampled frame
            if worst_frame is None:
                cap2 = cv2.VideoCapture(video_path)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
                ret, worst_frame = cap2.read()
                cap2.release()

            # Get guidance for worst severity
            guidance = self.get_guidance(highest_severity_cls, all_detections)

            return {
                "success": True,
                "processed_image": worst_frame,
                "detections": all_detections,
                "highest_severity": highest_severity_cls,
                "guidance": guidance,
                "timeline": timeline,
                "video_info": {
                    "duration": round(duration, 2),
                    "fps": round(video_fps, 1),
                    "frames_analyzed": frames_processed,
                    "total_incidents": len(all_detections)
                }
            }

        except Exception as e:
            return {"error": str(e)}

    def send_notification(self, detection_type, location, image_path=None):
        """Simulate email notification"""
        if detection_type not in AUTHORITY_CONTACTS:
            return False, "Invalid detection type"
            
        contact = AUTHORITY_CONTACTS[detection_type]
        
        # Simulation Logic
        print(f"--- SIMULATED EMAIL ---")
        print(f"To: {contact['email']}")
        print(f"Subject: {detection_type.upper()} REPORT at {location}")
        print("Body: Incident detected. Please respond.")
        print("-----------------------")
        
        return True, f"Notification sent to {contact['department']}"




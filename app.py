import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import tempfile
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import json
import os
from PIL import Image
import ollama
import base64
import requests

# ==================== CONFIG ====================
WEIGHTS_PATH = r"D:\python\citypulse\runs_citypulse\yolov8n_pothole_vbest2\weights\best.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50
LLM_MODEL_NAME = "mistral"

# Class mapping
CLASS_NAMES = {
    0: 'no_accident',
    1: 'minor_accident',
    2: 'moderate_accident',
    3: 'severe_accident',
    4: 'totaled_vehicle',
    5: 'pothole'
}

# Authority contacts
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

# ==================== MEDICAL GUIDANCE DATA ====================
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
            '✓ SAFETY FIRST: Turn on hazard lights, set up warning triangles 50-100 feet behind vehicle',
            '✓ ASSESS INJURIES: Check all occupants - Ask "Are you hurt? Any pain? Can you move everything?"',
            '✓ CALL AUTHORITIES: Dial 100 (Police) for official report - Required for insurance claims',
            '✓ SECURE EVIDENCE: Take 20+ photos - All angles of damage, license plates, road conditions, traffic signals',
            '✓ EXCHANGE INFO: Names, phone numbers, addresses, license numbers, insurance details, vehicle make/model',
            '✓ WITNESS DETAILS: Get contact info from any witnesses present',
            '✓ NOTE CONDITIONS: Write down time, weather, road conditions, traffic flow immediately'
        ],
        'recommendations': [
            '🩹 TREATING MINOR INJURIES:',
            '  • Cuts/Scrapes: Rinse with clean water for 2-3 minutes, pat dry with sterile gauze, apply antibiotic ointment, cover with bandage',
            '  • Bleeding: Apply direct pressure for 10 minutes without peeking, elevate above heart if possible',
            '  • Bruises: Ice pack wrapped in cloth for 15-20 min on/off cycles, elevate injured area',
            '  • Burns from airbag/seatbelt: Cool with running water for 10+ minutes, cover loosely with sterile gauze',
            '  • Glass embedded: Do NOT remove - cover and seek medical help',
            '',
            '⚕️ MEDICAL MONITORING:',
            '  • Document all injuries with photos immediately - bruises darken over 24-48 hours',
            '  • Whiplash symptoms appear 6-24 hours later: neck stiffness, headache, shoulder pain, dizziness',
            '  • Concussion warning signs: confusion, memory loss, nausea, sensitivity to light, unequal pupils',
            '  • Airbag injuries: Chemical burns on face/arms, chest soreness, temporary hearing loss',
            '  • Keep injury log: Note time, symptoms, severity (1-10 scale)',
            '  • Visit doctor within 24-48 hours EVEN IF feeling fine - internal injuries have delayed onset',
            '  • Request full body X-rays if any bone pain develops',
            '',
            '📋 LEGAL & INSURANCE STEPS:',
            '  • Write detailed incident statement while memory is fresh - include exact sequence of events',
            '  • Draw simple diagram showing vehicle positions, direction of travel, point of impact',
            '  • Do NOT discuss fault or liability with other party - stick to objective facts only',
            '  • Record exact words of what other driver says at scene',
            '  • Note if other driver shows signs of impairment: alcohol smell, slurred speech, erratic behavior',
            '  • Report to insurance within 24 hours - delays can void coverage',
            '  • Keep all receipts: medical bills, vehicle repairs, rental car, missed work',
            '  • Do NOT accept cash settlement at scene - unknown injuries may develop',
            '  • Get police report number and officer badge number',
            '',
            '🧠 PSYCHOLOGICAL CARE:',
            '  • Accident shock is normal - adrenaline masks pain and emotions',
            '  • Rest for 24 hours - avoid driving if feeling shaken',
            '  • Talk to someone about the incident - bottling up increases anxiety',
            '  • Watch for signs of PTSD: flashbacks, avoiding driving, nightmares, hypervigilance',
            '  • Consider counseling if anxiety persists beyond 2 weeks',
            '',
            '🚗 VEHICLE SAFETY:',
            '  • Have vehicle inspected even for minor damage - frame/alignment issues may not be visible',
            '  • Check all safety systems: airbags, seatbelts, lights, brakes',
            '  • Document pre-accident vehicle condition for insurance',
            '  • Request diminished value assessment - accidents reduce resale value'
        ],
        'emergency_level': 1,
        'warning_signs': [
            '🚨 SEEK IMMEDIATE MEDICAL ATTENTION IF:',
            'Headache that worsens over hours',
            'Neck or back pain that increases',
            'Dizziness, balance problems, or vertigo',
            'Nausea or vomiting',
            'Confusion, memory gaps, or disorientation',
            'Vision changes: blurriness, double vision, sensitivity to light',
            'Ringing in ears or hearing loss',
            'Numbness or tingling anywhere',
            'Difficulty concentrating or "foggy" feeling',
            'Mood changes: irritability, depression, anxiety',
            'Chest pain or difficulty breathing',
            'Abdominal pain or tenderness'
        ],
        'do_not_do': [
            '❌ NEVER leave accident scene before police arrive - Hit and run is criminal offense',
            '❌ NEVER admit fault or apologize - Can be used against you legally ("I\'m sorry" = admission)',
            '❌ NEVER sign documents without reading fully - Get lawyer review if unsure',
            '❌ NEVER accept immediate cash settlements - Unknown injuries may develop',
            '❌ NEVER discuss insurance limits with other party - They may claim maximum',
            '❌ NEVER give recorded statements to other party\'s insurance without lawyer',
            '❌ NEVER post about accident on social media - Can be used against you',
            '❌ NEVER say "I\'m fine" if you\'re not - Establish injury record immediately',
            '❌ NEVER move vehicles if there are ANY injuries - Preserve evidence',
            '❌ NEVER accept fault assignment from other driver - Police determine fault'
        ]
    },
    'moderate_accident': {
        'severity': 'Moderate',
        'immediate_actions': [
            '🚨 CALL 108 (AMBULANCE) AND 100 (POLICE) IMMEDIATELY - State exact location and number of injuries',
            '⚠️ SCENE SAFETY: Turn off all engines, check for fuel leaks (smell), sparks, smoke',
            '✓ DO NOT move injured persons unless immediate danger (fire, vehicle about to be hit)',
            '✓ PRIMARY ASSESSMENT: Check level of consciousness - "What\'s your name? Where are you? What day is it?"',
            '✓ AIRWAY CHECK: Ensure victim can breathe - Look for chest rise, listen for breath sounds',
            '✓ BREATHING: Count breaths per minute (normal: 12-20), note if rapid, shallow, or labored',
            '✓ CIRCULATION: Find pulse at wrist or neck, note strength and rate',
            '✓ BLEEDING CONTROL: Apply direct, firm pressure with cleanest cloth available for 10-15 minutes',
            '✓ SPINAL PRECAUTIONS: Keep head, neck, and back still - Place hands on both sides of head',
            '✓ SHOCK PREVENTION: Lay victim flat (unless breathing difficulty), cover with blanket/jacket',
            '✓ REASSURANCE: Talk calmly - "Help is coming, stay still, you\'re going to be okay"',
            '✓ MONITOR: Check pulse and breathing every 2-3 minutes until ambulance arrives'
        ],
        'recommendations': [
            '🩸 SEVERE BLEEDING CONTROL:',
            '  • Apply FIRM direct pressure - Use both hands if needed',
            '  • Do NOT peek to see if bleeding stopped - Disrupts clot formation',
            '  • If blood soaks through: Add MORE cloth on top, never remove original',
            '  • Elevate bleeding limb ABOVE heart level (if no fracture suspected)',
            '  • Pressure points as backup (if direct pressure insufficient):',
            '    - ARM: Brachial artery - Press inside of upper arm against bone',
            '    - LEG: Femoral artery - Press in groin crease where leg meets torso',
            '  • Time pressure: Note when you started - Tell paramedics exact duration',
            '  • If bleeding is from severed artery (spurting blood): Maintain maximum pressure, call for tourniquet',
            '',
            '🫀 SHOCK RECOGNITION & TREATMENT:',
            '  • Early signs: Pale/cool/clammy skin, rapid weak pulse, rapid shallow breathing, anxiety, thirst',
            '  • Late signs: Blue lips/nails, confusion, weakness, dilated pupils, loss of consciousness',
            '  • Position: Lay flat, elevate legs 12 inches (unless spinal injury or difficulty breathing)',
            '  • Temperature: Cover with blanket to prevent heat loss - Hypothermia worsens shock',
            '  • Do NOT give food/water - May vomit and choke, or need surgery',
            '  • Loosen tight clothing at neck, chest, waist',
            '  • Turn head to side if vomiting (while keeping neck still)',
            '  • Reassure continuously - Anxiety worsens shock',
            '',
            '🧠 HEAD INJURY PROTOCOL:',
            '  • Keep head and neck COMPLETELY still - Manual stabilization if must move',
            '  • Check pupils: Shine light in eyes - Both should constrict equally',
            '  • Unequal pupils = CRITICAL EMERGENCY = Brain bleeding',
            '  • Clear/bloody fluid from nose/ears = Skull fracture = Do NOT plug or clean',
            '  • Bump/bruise developing on head = Note location and size',
            '  • Level of consciousness: Check every 2-3 minutes:',
            '    - Alert: Knows name, place, date, what happened',
            '    - Verbal: Responds to questions but confused',
            '    - Pain: Only responds to painful stimulus',
            '    - Unresponsive: No response to any stimulus',
            '  • If condition worsens: Repeat emergency call with update',
            '',
            '🫁 BREATHING DIFFICULTIES:',
            '  • If conscious and breathing with difficulty: Help sit up in most comfortable position',
            '  • Support back and head - Leaning forward often easiest',
            '  • Loosen tight clothing around neck and chest',
            '  • Encourage slow, deep breaths - Count "breathe in 1-2-3, out 1-2-3"',
            '  • Watch for: Blue lips/nails, gasping, inability to speak full sentences',
            '  • If unconscious but breathing: Recovery position (if NO spinal injury):',
            '    1. Kneel beside victim',
            '    2. Place far arm at right angle',
            '    3. Bring near arm across chest',
            '    4. Bend far leg at knee',
            '    5. Roll toward you using leg leverage',
            '    6. Tilt head back slightly to keep airway open',
            '    7. Monitor breathing continuously',
            '',
            '💔 CHEST INJURIES:',
            '  • Broken ribs: Support chest with pillow or folded clothing',
            '  • Encourage shallow breathing if deep breaths cause severe pain',
            '  • Penetrating chest wound: Cover with plastic/aluminum foil taped on 3 sides',
            '    - Creates flutter valve: Air escapes but doesn\'t enter',
            '  • If victim coughing up blood: Position slightly elevated, lean toward injured side',
            '',
            '🦴 SUSPECTED FRACTURES:',
            '  • Immobilize injured area: Splint above AND below injury site',
            '  • Do NOT try to straighten or realign bones',
            '  • Check "CSM" distal to injury (fingers/toes):',
            '    - Circulation: Warm, pink color',
            '    - Sensation: Can feel light touch',
            '    - Movement: Can wiggle fingers/toes',
            '  • If CSM absent: Medical emergency - Nerve/vessel damage',
            '  • Makeshift splints: Rolled newspaper, magazines, sticks, pillows',
            '  • Pad splint with soft material to prevent pressure sores',
            '  • Secure with ties above, below, and across fracture site',
            '',
            '🤕 ABDOMINAL INJURIES:',
            '  • Rigid or swollen abdomen = Internal bleeding = Life-threatening',
            '  • Do NOT give food/water/medications',
            '  • Position: Lying on back with knees bent (reduces tension)',
            '  • If organs protruding: Cover with moist sterile cloth, do NOT push back in',
            '  • Impaled object: Stabilize in place with bulky dressings, do NOT remove',
            '',
            '📝 INFORMATION FOR PARAMEDICS:',
            '  • Mechanism of injury: Speed, point of impact, airbag deployment',
            '  • Initial condition vs current condition',
            '  • All treatments you provided',
            '  • Victim\'s medical history if known: Medications, allergies, conditions',
            '  • Time of injury and time of each status change',
            '  • Witness contact information'
        ],
        'emergency_level': 2,
        'warning_signs': [
            '🚨 CALL 108 AGAIN IMMEDIATELY IF CONDITION WORSENS:',
            'Decreasing level of consciousness (more confused, less responsive)',
            'Difficulty breathing or breathing rate changes significantly',
            'Chest pain or pressure',
            'Bleeding that won\'t stop after 15 minutes of firm pressure',
            'Suspected spinal injury: Neck/back pain, numbness, tingling, inability to move limbs',
            'Abdominal pain, rigidity, or swelling',
            'Signs of internal bleeding: Coughing blood, vomiting blood, blood in urine',
            'Pale, cold, clammy skin with rapid weak pulse (shock)',
            'Confused or disoriented behavior worsening',
            'Seizure or convulsions',
            'Severe headache that develops or worsens',
            'Clear fluid leaking from nose or ears (brain fluid)',
            'Unequal pupil size or pupils not reacting to light',
            'Loss of sensation or movement in any body part'
        ],
        'do_not_do': [
            '❌ Do NOT move victim unless IMMEDIATE life threat (fire, explosion, vehicle about to be struck)',
            '❌ Do NOT give food, water, or medication - May need surgery, could choke',
            '❌ Do NOT remove motorcycle/bicycle helmet unless airway is blocked',
            '❌ Do NOT try to push protruding organs back inside body',
            '❌ Do NOT remove impaled objects (glass, metal, debris)',
            '❌ Do NOT apply tourniquet unless bleeding absolutely cannot be controlled',
            '❌ Do NOT assume neck/spine is okay - Treat as injured until proven otherwise',
            '❌ Do NOT allow victim to smoke, eat, or drink',
            '❌ Do NOT leave victim unattended - Condition can deteriorate rapidly',
            '❌ Do NOT straighten fractured limbs',
            '❌ Do NOT pack nose bleeds - Could be skull fracture'
        ]
    },
    'severe_accident': {
        'severity': 'Severe',
        'immediate_actions': [
            '🚨🚨 CALL 108 - STATE "SEVERE ACCIDENT, LIFE THREATENING"',
            '⚠️ Check for fire, fuel leaks, electrical hazards',
            '✓ Quick assessment of all victims',
            '✓ START CPR if no breathing/pulse and trained',
            '✓ Control severe bleeding with maximum pressure',
            '✓ Keep victims still and warm'
        ],
        'recommendations': [
            '🔴 For severe bleeding: Apply maximum pressure with both hands',
            '🔴 CPR: 30 compressions, 2 breaths, continue until help arrives',
            '🔴 Spinal injury: Keep head and neck absolutely still',
            '🔴 Shock: Lay flat, elevate legs, keep warm'
        ],
        'emergency_level': 3,
        'warning_signs': [
            'No breathing or pulse',
            'Uncontrollable bleeding',
            'Unconscious or unresponsive',
            'Suspected spinal injury'
        ],
        'do_not_do': [
            'NEVER move victim except for fire/explosion',
            'NEVER remove impaled objects',
            'NEVER assume unconscious victim is okay'
        ]
    },
    'totaled_vehicle': {
        'severity': 'Critical',
        'immediate_actions': [
            '🚨🚨🚨 CALL 108 - "CRITICAL MULTI-CASUALTY ACCIDENT"',
            '⚠️ Check for fuel leaks, fire, electrical hazards',
            '⚠️ If fire/explosion imminent: Move victims immediately',
            '✓ Triage multiple victims: Prioritize most critical',
            '✓ Start CPR on victims without pulse',
            '✓ Control catastrophic bleeding'
        ],
        'recommendations': [
            '🔴🔴 Cardiac arrest: Hard, fast compressions 100-120/min',
            '🔴🔴 Catastrophic bleeding: Maximum pressure or tourniquet',
            '🔴🔴 Fire risk: Evacuate all victims immediately',
            '🔴🔴 Multiple casualties: Triage and prioritize RED cases'
        ],
        'emergency_level': 4,
        'warning_signs': [
            'No breathing or pulse',
            'Catastrophic bleeding',
            'Fire or explosion imminent',
            'Multiple critical injuries'
        ],
        'do_not_do': [
            'NEVER move victim except immediate life threat',
            'NEVER leave critically injured unattended'
        ]
    },
    'pothole': {
        'severity': 'Infrastructure',
        'immediate_actions': [
            '✓ Note exact location with GPS coordinates',
            '✓ Document with photos showing size and depth',
            '✓ Report to municipal authorities immediately',
            '✓ Warn other drivers if safe to do so'
        ],
        'recommendations': [
            '• Take photos from multiple angles',
            '• Measure approximate dimensions if safe',
            '• Note any damage to your vehicle',
            '• Report through official municipal app/website'
        ],
        'emergency_level': 0,
        'warning_signs': [],
        'do_not_do': [
            'Do NOT attempt repairs yourself',
            'Do NOT create additional hazards'
        ]
    }
}

# ==================== LLM GUIDANCE FUNCTION ====================
def get_llm_guidance(severity_class, detection_details):
    """Get AI-enhanced emergency guidance using Ollama LLM."""
    guidance = ACCIDENT_GUIDANCE.get(severity_class, ACCIDENT_GUIDANCE['no_accident'])
    
    try:
        system_prompt = """You are an emergency medical advisor. Provide ONLY the top 3 most critical, 
life-saving actions a civilian should take immediately. Be extremely concise - maximum 3 short bullet points."""
        
        prompt = f"""Incident detected: {severity_class.replace('_', ' ')}
        
List the 3 most CRITICAL immediate actions to save lives right now."""
        
        response = ollama.generate(
            model=LLM_MODEL_NAME,
            prompt=prompt,
            system=system_prompt,
            options={
                'temperature': 0.1,
                'num_ctx': 2048
            }
        )
        
        llm_text = response.get('response', 'Could not generate AI summary.')
        guidance['llm_enhanced'] = llm_text.strip()
        
    except Exception as e:
        guidance['llm_enhanced'] = (
            f"⚠️ LLM unavailable: {e.__class__.__name__}. "
            f"Ensure Ollama is running with model '{LLM_MODEL_NAME}'."
        )
    
    return guidance

# ==================== EMAIL NOTIFICATION ====================
def send_notification_email(incident_type, location, image_data=None):
    """
    Simulate sending emergency notification email.
    In production, replace with actual API calls to emergency services.
    """
    try:
        contact = AUTHORITY_CONTACTS.get(incident_type, AUTHORITY_CONTACTS['accident'])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        subject = f"🚨 URGENT: {incident_type.upper()} Detected at {location}"
        body = f"""
AUTOMATED INCIDENT REPORT
========================
Time: {timestamp}
Location: {location}
Incident Type: {incident_type.upper()}
Department: {contact['department']}
Emergency Contact: {contact['phone']}

This is an automated detection. Please respond immediately.

[In production: This would include image attachments and GPS coordinates]
        """
        
        # DEMO MODE - Just log the notification
        st.info(f"📧 **Notification Sent (Simulated)**\n\nTo: {contact['email']}\nSubject: {subject}")
        
        return True, f"✅ Emergency notification sent to {contact['department']}"
        
    except Exception as e:
        return False, f"❌ Failed to send notification: {str(e)}"

# ==================== YOLO DETECTION CORE ====================
@st.cache_resource
def load_model():
    """Load the YOLOv8 model."""
    try:
        model = YOLO(WEIGHTS_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_device():
    """Determine the device to use."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def detect_on_image(img_array, model, device):
    """Run detection on a single image."""
    try:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        results = model(
            img_bgr, 
            conf=CONF_THRESHOLD, 
            iou=IOU_THRESHOLD, 
            device=device,
            verbose=False
        )[0]
        return results
    except Exception as e:
        st.error(f"Detection error: {e}")
        return None

def draw_detections(img_bgr, results):
    """Draw bounding boxes and labels on the image."""
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf.item())
        cls_id = int(box.cls.item())
        cls_name = CLASS_NAMES.get(cls_id, 'Unknown')
        
        # Color coding
        if cls_name == 'no_accident':
            color = (0, 255, 0)
        elif 'minor' in cls_name:
            color = (0, 255, 255)
        elif 'moderate' in cls_name:
            color = (0, 165, 255)
        elif 'severe' in cls_name or 'totaled' in cls_name:
            color = (0, 0, 255)
        elif 'pothole' in cls_name:
            color = (255, 255, 0)
        else:
            color = (255, 0, 255)

        label = f"{cls_name} {conf:.2f}"
        
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_bgr, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def detect_on_video(uploaded_file, model, device, detection_placeholder):
    """Process uploaded video frame by frame."""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    results = None
    frame_count = 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    st.info(f"Processing video @ {fps:.2f} FPS...")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect every 3 frames for performance
            if frame_count % 3 == 0:
                yolo_results = model(
                    frame, 
                    conf=CONF_THRESHOLD, 
                    iou=IOU_THRESHOLD, 
                    device=device,
                    verbose=False
                )
                if yolo_results and len(yolo_results) > 0:
                    results = yolo_results[0]
            
            # Draw annotations
            if results is not None:
                annotated_frame = draw_detections(frame.copy(), results)
                caption = f"Frame {frame_count}"
            else:
                annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                caption = f"Frame {frame_count} - No detections yet"
                
            detection_placeholder.image(
                annotated_frame, 
                channels="RGB", 
                caption=caption, 
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Video processing error: {e}")
        results = None
        
    finally:
        cap.release()
        try:
            os.unlink(video_path)
        except:
            pass
            
    return results

# ==================== STREAMLIT APP ====================
def main():
    st.set_page_config(
        page_title="Intelligent Road Incident Detection System",
        page_icon="🚨",
        layout="wide"
    )

    st.title("🚨 Intelligent Road Incident Detection & First Aid System")
    st.markdown("AI-powered detection of road incidents with real-time emergency guidance")
    
    model = load_model()
    device = get_device()
    if model is None:
        st.error("Failed to load model. Check WEIGHTS_PATH configuration.")
        return
    
    st.sidebar.success(f"Model loaded on {device.upper()}")

    # Sidebar Configuration
    st.sidebar.header("Configuration")
    
    global CONF_THRESHOLD, IOU_THRESHOLD 
    CONF_THRESHOLD = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, CONF_THRESHOLD, 0.05)
    IOU_THRESHOLD = st.sidebar.slider("IoU Threshold", 0.0, 1.0, IOU_THRESHOLD, 0.05)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Reporting")
    location = st.sidebar.text_input("Incident Location", "NH48, Pune, India")
    notify_authorities = st.sidebar.checkbox("Simulate Emergency Notification", True)

    # Main Content
    st.header("Upload Media for Analysis")
    upload_type = st.radio("Input Type:", ("Image", "Video"), horizontal=True)

    results = None

    if upload_type == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                with st.spinner("Analyzing..."):
                    results = detect_on_image(img_array, model, device)
                    if results is not None:
                        annotated_img = draw_detections(img_array.copy(), results)
                        st.subheader("Detection Results")
                        st.image(annotated_img, channels="RGB", use_container_width=True)
            
    elif upload_type == "Video":
        uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])
        
        if uploaded_file is not None:
            st.subheader("Real-time Analysis")
            detection_placeholder = st.empty()
            results = detect_on_video(uploaded_file, model, device, detection_placeholder)

    # Post-Detection Analysis
    if results is not None and results.boxes is not None and len(results.boxes) > 0:
        st.markdown("---")
        st.subheader("Incident Summary")
        
        detections = []
        highest_severity_cls = 'no_accident'
        highest_emergency_level = 0
        
        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            cls_name = CLASS_NAMES.get(cls_id, 'unknown')
            
            if cls_name != 'no_accident':
                detections.append({'class': cls_name, 'confidence': conf})
                current_level = ACCIDENT_GUIDANCE.get(cls_name, {'emergency_level': 0})['emergency_level']
                if current_level > highest_emergency_level:
                    highest_emergency_level = current_level
                    highest_severity_cls = cls_name
        
        if not detections:
            st.success("No incidents detected")
            return

        # Display Alert
        if highest_severity_cls == 'pothole':
            st.warning(f"🚧 Pothole Detected")
        else:
            st.error(f"🚨 {highest_severity_cls.replace('_', ' ').title()}")
        
        # Emergency Guidance
        st.markdown("---")
        st.subheader("Emergency Guidance")
        guidance = get_llm_guidance(highest_severity_cls, detections)

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### Immediate Actions ({guidance['severity']})")
            for action in guidance['immediate_actions']:
                st.markdown(f"**{action}**")

        with col2:
            contact_type = 'pothole' if highest_severity_cls == 'pothole' else 'accident'
            st.markdown(f"### Emergency Contacts")
            st.markdown(f"**Department:** {AUTHORITY_CONTACTS[contact_type]['department']}")
            st.markdown(f"**Phone:** {AUTHORITY_CONTACTS[contact_type]['phone']}")
            
            if guidance['warning_signs']:
                st.markdown("---")
                st.markdown("### Warning Signs")
                for sign in guidance['warning_signs'][:5]:
                    st.markdown(f"- {sign}")
                        
        # Detailed Guidance
        with st.expander("Detailed Recommendations", expanded=False):
            for rec in guidance['recommendations']:
                st.markdown(f"- {rec}")
            
            st.markdown("---")
            st.markdown("### DO NOT DO")
            for item in guidance['do_not_do']:
                st.markdown(f"**{item}**")

        # LLM Enhanced Summary
        if 'llm_enhanced' in guidance:
            st.markdown("---")
            st.subheader("AI Quick Summary")
            st.info(guidance['llm_enhanced'])

        # Send Notification
        st.markdown("---")
        if notify_authorities and location:
            contact_type = 'pothole' if highest_severity_cls == 'pothole' else 'accident'
            success, message = send_notification_email(contact_type, location)
            if success:
                st.success(message)
            else:
                st.warning(message)
        elif notify_authorities:
            st.warning("Please provide location to send notification")

if __name__ == "__main__":
    main()
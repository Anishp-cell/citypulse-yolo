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
WEIGHTS_PATH = r"D:\python\citypulse\runs_citypulse\yolov8n_pothole_vbest2\weights\best.pt"  # Update with your model path
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

# Authority contacts (Demo - replace with actual contact system in production)
AUTHORITY_CONTACTS = {
    'pothole': {
        'email': 'demo.roads.dept@example.com',  # Demo email - use actual municipal API/system
        'phone': '100 (Municipal Helpline)',
        'department': 'Roads & Infrastructure Department'
    },
    'accident': {
        'email': 'demo.emergency@example.com',  # Demo email - use actual emergency API
        'phone': '108 (Ambulance) / 100 (Police)',
        'department': 'Emergency Services'
    }
}

# ==================== MEDICAL GUIDANCE DATA (Previously provided and included) ====================
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
            '✓ Ensure safety - Turn on hazard lights and set up warning triangles',
            '✓ Check all occupants for injuries - Ask about pain, dizziness, nausea, or discomfort',
            '✓ Move vehicles to shoulder if safe and drivable to avoid blocking traffic',
            '✓ Call 100 (Police) for accident report - Required for insurance claims',
            '✓ Exchange information: Names, phone numbers, license plates, insurance details',
            '✓ Document scene thoroughly: Take photos from multiple angles of damage, license plates, road conditions',
            '✓ Note weather conditions, traffic signals, and any witnesses',
            '✓ Do NOT admit fault - Stick to facts only'
        ],
        'recommendations': [
            '• Cuts/Scrapes: Rinse with clean water, apply pressure if bleeding, cover with sterile bandage',
            '• Bruises: Apply ice pack wrapped in cloth for 15-20 minutes to reduce swelling',
            '• Minor Pain: Note location and intensity - avoid taking pain medication until assessed by doctor',
            '• Whiplash symptoms may appear 6-24 hours later: neck stiffness, headache, dizziness',
            '• Concussion warning: Watch for confusion, memory loss, nausea, unequal pupils',
            '• Airbag injuries: Check for burns on face/arms, chest discomfort, hearing issues',
            '• Photograph any visible injuries immediately',
            '• Keep all medical records and receipts for insurance',
            '• Visit doctor within 24-48 hours even if feeling fine - some injuries have delayed onset',
            '• Rest for 24 hours and avoid driving if feeling shaken'
        ],
        'emergency_level': 1,
        'warning_signs': [
            'Increasing headache or neck pain',
            'Dizziness or balance problems',
            'Nausea or vomiting',
            'Confusion or memory problems',
            'Vision changes or ringing in ears',
            'Any numbness or tingling'
        ],
        'do_not_do': [
            'NEVER leave accident scene before police arrive (hit and run charges)',
            'NEVER sign documents without reading and understanding fully',
            'NEVER accept immediate cash settlements',
            'NEVER discuss insurance limits with other party',
            'NEVER admit fault or apologize (can be used against you)',
            'NEVER move vehicles if there are injuries',
            'NEVER give recorded statements without lawyer present'
        ]
    },
    'moderate_accident': {
        'severity': 'Moderate',
        'immediate_actions': [
            '🚨 CALL 108 (AMBULANCE) AND 100 (POLICE) IMMEDIATELY',
            '✓ Scene safety first - Turn off all vehicles if possible, check for fuel leaks',
            '✓ DO NOT move injured persons unless vehicle is on fire or about to be hit',
            '✓ Check level of consciousness - Ask name, date, location. Note any confusion',
            '✓ Check breathing - Look for chest rise, listen for breath sounds, feel for air',
            '✓ Control bleeding - Apply direct pressure with cleanest cloth available',
            '✓ Cover open wounds with sterile/clean cloth to prevent contamination',
            '✓ Keep injured person still - Especially head, neck, and back',
            '✓ Monitor vital signs - Check pulse at wrist or neck, count breaths per minute',
            '✓ Keep victim warm - Cover with blanket, jacket, or emergency blanket',
            '✓ Provide reassurance - Speak calmly, explain help is coming',
            '✓ If unconscious but breathing - Maintain airway, prepare to place in recovery position if vomiting'
        ],
        'recommendations': [
            '• BLEEDING CONTROL: Apply firm direct pressure for 10-15 minutes without checking if bleeding stopped',
            '• If blood soaks through: Add more cloth on top, do NOT remove original cloth',
            '• Elevate bleeding limb above heart level if no fracture suspected',
            '• SHOCK PREVENTION: Lay victim flat, elevate legs 12 inches (if no spinal injury suspected)',
            '• Signs of shock: Pale/clammy skin, rapid weak pulse, rapid breathing, confusion, thirst',
            '• HEAD INJURY: Keep head still, watch for clear fluid from nose/ears (brain fluid), unequal pupils',
            '• CHEST INJURY: If difficulty breathing, help victim sit up slightly if no spinal injury',
            '• ABDOMINAL INJURY: Do not give food/water, watch for rigidity or swelling',
            '• FRACTURES: Immobilize injured area, do not try to realign bones',
            '• If victim is cold: Cover with blankets but avoid overheating',
            '• If victim is confused: Reorient gently, note time confusion started',
            '• RECOVERY POSITION (if unconscious but breathing):',
            '  - Place victim on their side',
            '  - Tilt head back slightly to maintain airway',
            '  - Position top leg bent at hip and knee for stability',
            '  - Monitor breathing continuously',
            '• Write down time of injury and vital signs for paramedics',
            '• Stay with victim - continuous monitoring is critical'
        ],
        'emergency_level': 2,
        'warning_signs': [
            'Decreasing level of consciousness',
            'Difficulty breathing or rapid breathing',
            'Chest pain or pressure',
            'Bleeding that won\'t stop after 15 min pressure',
            'Suspected spinal injury (neck/back pain, numbness, tingling)',
            'Abdominal pain or rigidity',
            'Signs of internal bleeding (coughing blood, blood in vomit)',
            'Pale, cold, clammy skin',
            'Confused or disoriented behavior',
            'Seizures'
        ],
        'do_not_do': [
            'Do NOT move victim unless absolutely necessary (fire/explosion risk)',
            'Do NOT give food, water, or medication',
            'Do NOT remove helmet if motorcyclist (unless airway blocked)',
            'Do NOT try to push protruding organs back in',
            'Do NOT remove impaled objects',
            'Do NOT apply tourniquet unless bleeding cannot be controlled',
            'Do NOT assume neck/spine is okay - treat as injured until proven otherwise'
        ]
    },
    'severe_accident': {
        'severity': 'Severe',
        'immediate_actions': [
            '🚨🚨 CALL 108 IMMEDIATELY - STATE "SEVERE ACCIDENT, LIFE THREATENING"',
            '⚠️ SCENE SAFETY - Check for fire, fuel, electrical hazards, unstable vehicles',
            '⚠️ DO NOT MOVE VICTIMS - Assume spinal injury until proven otherwise',
            '✓ Quick assessment - Count victims, identify most critical',
            '✓ Airway check - Tilt head back gently if no neck injury suspected, look/listen/feel for breathing',
            '✓ Breathing check - Place hand on chest, count breaths per minute (normal: 12-20)',
            '✓ Circulation check - Find pulse at neck (carotid) or wrist, note strength and rate',
            '✓ START CPR if no breathing/pulse and trained - Every second counts',
            '✓ Severe bleeding control - Apply maximum direct pressure, use multiple cloth layers',
            '✓ Spinal stabilization - One person holds head still in neutral position if victim must be moved',
            '✓ Cover open wounds - Use cleanest material available to prevent infection',
            '✓ Keep victim warm - Hypothermia worsens shock',
            '✓ Talk to conscious victims - Keep them calm and still, explain what you\'re doing',
            '✓ Clear bystanders - Assign specific tasks: call ambulance, direct traffic, bring first aid kit',
            '✓ Note time - Record when injuries occurred and when help was called'
        ],
        'recommendations': [
            '🔴 SEVERE BLEEDING (HEMORRHAGE):',
            '  - Apply direct pressure with both hands if needed',
            '  - Pack wound with gauze/cloth if deep',
            '  - Do NOT remove cloth if soaked - add more on top',
            '  - Apply pressure to pressure points if direct pressure fails:',
            '    * Arm: Brachial artery (inside of upper arm)',
            '    * Leg: Femoral artery (groin area)',
            '  - Tourniquet ONLY as absolute last resort for limb amputation/severe limb bleeding:',
            '    * Place 2-3 inches above wound',
            '    * Tighten until bleeding stops',
            '    * Note exact time applied',
            '    * Do NOT loosen until medical help arrives',
            '',
            '🔴 NOT BREATHING - CPR PROTOCOL:',
            '  1. Check scene safety first',
            '  2. Tap shoulders, shout "Are you okay?"',
            '  3. Call for help/call 108',
            '  4. Tilt head back, lift chin (if no neck injury)',
            '  5. Look/listen/feel for breathing (max 10 seconds)',
            '  6. If not breathing: Start CPR immediately',
            '  7. Hand position: Center of chest, between nipples',
            '  8. Push hard and fast: 2 inches deep, 100-120 per minute',
            '  9. 30 compressions, then 2 rescue breaths',
            '  10. Continue until help arrives or victim starts breathing',
            '',
            '🔴 SUSPECTED SPINAL INJURY:',
            '  - Immobilize head and neck immediately',
            '  - One person maintains in-line stabilization',
            '  - Do NOT tilt, rotate, or flex neck',
            '  - If must move (fire risk): Log roll with 3+ people keeping spine aligned',
            '  - Look for: Neck/back pain, numbness, tingling, inability to move limbs',
            '',
            '🔴 SHOCK MANAGEMENT:',
            '  - Lay flat (if no spinal/chest injury)',
            '  - Elevate legs 12 inches',
            '  - Keep warm with blankets',
            '  - Do NOT give anything by mouth',
            '  - Monitor vital signs every 2-3 minutes',
            '  - Signs: Pale/blue skin, weak rapid pulse, shallow breathing, confusion, cold/clammy',
            '',
            '🔴 HEAD TRAUMA:',
            '  - Keep head and neck completely still',
            '  - Monitor consciousness level closely',
            '  - Look for: Clear/bloody fluid from ears/nose (brain fluid), unequal pupils',
            '  - If vomiting: Turn whole body as unit to side (maintain spine alignment)',
            '',
            '🔴 CHEST INJURIES:',
            '  - Sucking chest wound: Cover with plastic taped on 3 sides (allows air out, not in)',
            '  - Broken ribs: Support chest with pillow/clothing',
            '  - Difficulty breathing: Help victim sit up into most comfortable position',
            '',
            '🔴 ABDOMINAL INJURIES:',
            '  - Exposed organs: Cover with moist sterile cloth, do NOT push back in',
            '  - Impaled object: Stabilize in place, do NOT remove',
            '  - Look for: Rigidity, swelling, bruising',
            '',
            '🔴 BURNS (if vehicle fire):',
            '  - Remove from heat source',
            '  - Cool with water (not ice) for 10-20 minutes',
            '  - Do NOT remove stuck clothing',
            '  - Cover with clean, dry cloth',
            '  - Do NOT apply ointments',
            '',
            '🔴 FRACTURES:',
            '  - Immobilize above and below injury site',
            '  - Do NOT try to straighten or realign',
            '  - Check pulse and sensation below injury',
            '  - Splint with rigid materials if available',
            '',
            '• Assign roles to bystanders:',
            '  - Person 1: Call emergency services',
            '  - Person 2: Control traffic/scene safety',
            '  - Person 3: Get first aid kit, blankets',
            '  - Person 4: Take notes for paramedics',
            '',
            '• Document for paramedics:',
            '  - Time of accident',
            '  - Initial condition of victims',
            '  - Changes in condition',
            '  - Treatment provided',
            '  - Medications/allergies if known'
        ],
        'emergency_level': 3,
        'warning_signs': [
            'CRITICAL - No breathing or pulse',
            'CRITICAL - Uncontrollable bleeding',
            'CRITICAL - Unconscious or unresponsive',
            'CRITICAL - Not breathing normally',
            'Blue lips or fingernails (lack of oxygen)',
            'Severe chest pain or pressure',
            'Coughing or vomiting blood',
            'Severe head injury with confusion',
            'Seizures',
            'Suspected spinal injury',
            'Penetrating injuries to head/chest/abdomen',
            'Rapid deterioration of condition',
            'Signs of internal bleeding',
            'Difficulty speaking or slurred speech'
        ],
        'do_not_do': [
            'NEVER move victim except for immediate life threat (fire/explosion)',
            'NEVER give food, water, or medications',
            'NEVER remove impaled objects',
            'NEVER remove motorcycle/bike helmet unless airway blocked',
            'NEVER try to push protruding organs back inside',
            'NEVER apply tourniquet unless truly life-threatening and uncontrollable bleeding',
            'NEVER leave critically injured victim unattended',
            'NEVER assume victim is okay if unconscious then wakes up - brain injury can worsen',
            'NEVER bend, twist, or flex the spine if spinal injury suspected'
        ]
    },
    'totaled_vehicle': {
        'severity': 'Critical',
        'immediate_actions': [
            '🚨🚨🚨 CALL 108 NOW - SAY "CRITICAL MULTI-CASUALTY ACCIDENT"',
            '⚠️ IMMEDIATE HAZARD CHECK:',
            '  - Fuel leaks (smell of gasoline) - NO SMOKING, turn off phones',
            '  - Fire or smoke - Vehicle fire extinguisher (aim at base of flames)',
            '  - Electrical hazards - Downed power lines (stay away, call utility company)',
            '  - Unstable vehicles - May tip, roll, or explode',
            '  - Traffic - Set up warning perimeter 100+ feet',
            '⚠️ EVACUATION DECISION:',
            '  - If fire/explosion imminent: Move victims immediately (accept spinal risk)',
            '  - If no immediate hazard: DO NOT MOVE - wait for fire department',
            '✓ RAPID TRIAGE (if multiple victims):',
            '  - Walk-wounded (minor injuries): Move to safe area, self-monitor',
            '  - Critical (not breathing, severe bleeding): Prioritize',
            '  - Expectant (severe injuries, low survival chance): Comfort, monitor',
            '✓ ABC ASSESSMENT (most critical victim first):',
            '  - Airway: Open if blocked (careful of neck)',
            '  - Breathing: Look, listen, feel - start CPR if needed',
            '  - Circulation: Check pulse, control bleeding',
            '✓ CRUSH INJURIES - DO NOT MOVE if limb trapped >15 minutes (risk of crush syndrome)',
            '✓ ENTRAPMENT - Stabilize vehicle, support victim, wait for fire department',
            '✓ SPINAL STABILIZATION - Manual in-line if must move',
            '✓ ASSIGN BYSTANDERS:',
            '  - Person 1: Call 108, update with details',
            '  - Person 2: Flag down additional help',
            '  - Person 3: Control traffic 100 feet away',
            '  - Person 4: Get fire extinguisher, first aid kit',
            '  - Person 5: Document everything',
            '✓ START MOST CRITICAL TREATMENT:',
            '  - CPR if no pulse (continuous until paramedics arrive)',
            '  - Direct pressure on severe bleeding',
            '  - Airway management',
            '  - Shock prevention'
        ],
        'recommendations': [
            '🔴🔴 CARDIAC ARREST - CPR (if no pulse/not breathing):',
            '  1. Confirm scene safety',
            '  2. Place on firm surface (ground, not seat)',
            '  3. Hand position: Center of chest, between nipples',
            '  4. Lock elbows, position shoulders over hands',
            '  5. Compress HARD and FAST:',
            '     - Depth: At least 2 inches (5 cm)',
            '     - Rate: 100-120 compressions per minute',
            '     - Rhythm: "Stayin\' Alive" by Bee Gees',
            '  6. Allow full chest recoil between compressions',
            '  7. 30 compressions : 2 rescue breaths ratio',
            '  8. Rescue breaths (if trained):',
            '     - Tilt head back, lift chin',
            '     - Pinch nose shut',
            '     - Full seal over mouth',
            '     - Breath until chest rises (1 second)',
            '  9. Continue non-stop until:',
            '     - Victim starts breathing',
            '     - AED/defibrillator arrives',
            '     - Paramedics take over',
            '     - You are physically unable to continue',
            '  10. If AED available: Turn on, follow voice prompts',
            '  NOTE: Hands-only CPR acceptable if untrained in rescue breaths',
            '',
            '🔴🔴 CATASTROPHIC BLEEDING:',
            '  - Life-threatening bleeding must be stopped within 3-5 minutes',
            '  - Apply MAXIMUM direct pressure with both hands',
            '  - Pack deep wounds with gauze/clean cloth',
            '  - If bleeding continues through cloth: Add more, do NOT remove',
            '  - TOURNIQUET (last resort for limb bleeding):',
            '    * Commercial tourniquet preferred',
            '    * Improvised: Belt, cloth at least 2 inches wide',
            '    * Place 2-3 inches above wound (between wound and heart)',
            '    * Tighten until bleeding completely stops',
            '    * Secure tightly, do NOT loosen',
            '    * Mark forehead with "T" and exact time',
            '    * Write time on tourniquet itself',
            '    * Maximum safe time: 2 hours',
            '  - Hemostatic gauze: Pack into wound, hold pressure 3 minutes',
            '  - Monitor for continued internal bleeding: Pale skin, rapid pulse, confusion',
            '',
            '🔴🔴 FIRE/EXPLOSION RISK:',
            '  - FUEL LEAK:',
            '    * Eliminate all ignition sources immediately',
            '    * No phones, no smoking within 100 feet',
            '    * Evacuate everyone from vehicle',
            '    * Move victims upwind and at least 100 feet away',
            '    * Have bystanders call fire department',
            '  - ACTIVE FIRE:',
            '    * Assess if safe to attempt rescue (2-3 minutes max before full engulfment)',
            '    * If victim conscious: Guide them to self-evacuate',
            '    * If unconscious and fire small: Rapid extraction with spinal precautions',
            '    * If fire large: Do NOT attempt - severe injury to rescuer likely',
            '    * Use fire extinguisher: PASS method',
            '      P - Pull pin',
            '      A - Aim at base of fire',
            '      S - Squeeze handle',
            '      S - Sweep side to side',
            '  - POST-FIRE BURNS:',
            '    * Remove from heat source',
            '    * Do NOT remove stuck clothing',
            '    * Cool with water (not ice) for 10-20 minutes',
            '    * Cover with clean, dry cloth',
            '    * Treat for shock',
            '',
            '🔴🔴 TRAPPED VICTIMS:',
            '  - DO NOT attempt extraction unless:',
            '    * Fire/explosion imminent',
            '    * You are trained in vehicle extrication',
            '    * Have proper equipment',
            '  - If must extract:',
            '    * Minimum 3 people',
            '    * One person maintains manual inline spinal stabilization',
            '    * Log roll technique (keep spine aligned)',
            '    * Support head, neck, torso as single unit',
            '  - While waiting for fire department:',
            '    * Stabilize vehicle with chocks/rocks',
            '    * Turn off ignition if accessible',
            '    * Talk to victim, provide reassurance',
            '    * Monitor breathing and consciousness',
            '    * Have bystander stay with victim at all times',
            '',
            '🔴🔴 CRUSH INJURIES:',
            '  - Limb trapped >15 minutes: DO NOT MOVE',
            '  - Releasing crush can cause crush syndrome:',
            '    * Toxic substances released into bloodstream',
            '    * Can cause cardiac arrest when released',
            '    * Requires immediate medical intervention',
            '  - If must release (fire risk):',
            '    * Warn victim they may lose consciousness',
            '    * Be ready to start CPR immediately',
            '    * Apply tourniquet BEFORE releasing if possible',
            '  - Document exact time limb was trapped',
            '',
            '🔴🔴 MULTIPLE CASUALTIES - TRIAGE:',
            '  - Use START triage system:',
            '    * GREEN (Walking wounded): Minor injuries, can walk',
            '    * YELLOW (Delayed): Serious but stable, can wait 30-60 min',
            '    * RED (Immediate): Life-threatening, needs immediate care',
            '    * BLACK (Expectant): Severe injuries, unlikely to survive',
            '  - Priority order:',
            '    1. RED - Severe bleeding, airway compromise, shock',
            '    2. YELLOW - Fractures, moderate injuries',
            '    3. GREEN - Minor cuts, bruises, psychological',
            '  - Mark victims with triage tags if available',
            '  - Assign one person to track all victims',
            '',
            '🔴🔴 PENETRATING INJURIES:',
            '  - Impaled objects (metal, glass, debris):',
            '    * NEVER remove object',
            '    * Stabilize object in place with bulky dressings',
            '    * Control bleeding around object',
            '    * If object interferes with CPR: Trained personnel may need to remove',
            '  - Evisceration (organs protruding):',
            '    * Do NOT push organs back in',
            '    * Cover with moist sterile dressing',
            '    * Cover dressing with plastic wrap',
            '    * Keep victim still',
            '',
            '🔴🔴 TRAUMATIC BRAIN INJURY:',
            '  - Keep head and neck absolutely still',
            '  - Monitor consciousness every 2-3 minutes',
            '  - Look for:',
            '    * Clear or bloody fluid from ears/nose (CSF leak)',
            '    * Unequal pupil size',
            '    * Confusion, combativeness',
            '    * Seizures',
            '    * Decreasing level of consciousness',
            '  - If victim vomits: Log roll entire body to side',
            '',
            '🔴🔴 EMERGENCY CHILDBIRTH (if pregnant victim):',
            '  - If crowning visible and delivery imminent:',
            '    * Support head as it emerges',
            '    * Check for cord around neck (gently loop over head)',
            '    * Support body as it delivers',
            '    * Dry and warm baby immediately',
            '    * Stimulate baby to cry (rub back)',
            '    * Keep baby skin-to-skin with mother',
            '  - Do NOT pull on cord',
            '  - Do NOT cut cord without sterile equipment',
            '',
            '• SCENE ORGANIZATION:',
            '  - Designate incident commander',
            '  - Create treatment area (safe, away from wreckage)',
            '  - Stage area for arriving ambulances',
            '  - Keep crowd back minimum 100 feet',
            '  - Document everything: Times, treatments, victim status changes',
            '',
            '• STRESS MANAGEMENT FOR RESPONDERS:',
            '  - This is traumatic for everyone',
            '  - Focus on one task at a time',
            '  - Accept that you cannot save everyone',
            '  - Seek critical incident stress debriefing after',
            '  - Know that doing something is better than doing nothing'
        ],
        'emergency_level': 4,
        'warning_signs': [
            'IMMEDIATE LIFE THREAT - No breathing or pulse',
            'IMMEDIATE LIFE THREAT - Catastrophic bleeding (spurting blood)',
            'IMMEDIATE LIFE THREAT - Fire or explosion imminent',
            'IMMEDIATE LIFE THREAT - Airway obstruction',
            'Unconscious or unresponsive',
            'Severe difficulty breathing or no breathing',
            'Chest wound sucking air',
            'Severe head trauma with decreasing consciousness',
            'Uncontrollable bleeding despite pressure',
            'Signs of internal bleeding (rigid abdomen, coughing blood)',
            'Suspected spinal injury with numbness/paralysis',
            'Crush injuries to torso or multiple limbs',
            'Traumatic amputation',
            'Severe burns to face, chest, or large body areas',
            'Penetrating trauma to head, chest, or abdomen',
            'Multiple injuries to same victim',
            'Rapid deterioration of vital signs'
        ],
        'do_not_do': [
            'NEVER move victim except for immediate life threat (fire/explosion)',
            'NEVER give food, water, or medications',
            'NEVER remove impaled objects',
            'NEVER remove motorcycle/bike helmet unless airway blocked',
            'NEVER try to push protruding organs back inside',
            'NEVER apply tourniquet unless truly life-threatening and uncontrollable bleeding',
            'NEVER leave critically injured victim unattended',
            'NEVER assume victim is okay if unconscious then wakes up - brain injury can worsen',
            'NEVER bend, twist, or flex the spine if spinal injury suspected'
        ]
    }
}

# ==================== LLM GUIDANCE FUNCTION ====================

# Assuming LLM_MODEL_NAME is defined elsewhere (e.g., LLM_MODEL_NAME = "mistral")
# Assuming ACCIDENT_GUIDANCE is defined elsewhere

def get_llm_guidance(severity_class, detection_details):
    # Fallback to structured guidance (your existing logic)
    guidance = ACCIDENT_GUIDANCE.get(severity_class, ACCIDENT_GUIDANCE['no_accident'])
    
    # --- LLM Enhancement ---
    try:
        # 1. Define the system prompt for role-playing
        system_prompt = "You are an emergency medical advisor and triage specialist. Your output must be extremely concise and actionable, focusing only on the most critical immediate steps (maximum 4 bullet points)."
        
        # 2. Define the user prompt using the detection context
        prompt = f'''
        Based on the highest detected incident: {severity_class}
        List the 3 most CRITICAL, immediate, life-saving actions a civilian should take right now.
        '''
        
        # 3. Call the Ollama API
        response = ollama.generate(
            model=LLM_MODEL_NAME, # Uses the model name from your config
            prompt=prompt,
            system=system_prompt,
            options={
                'temperature': 0.1, # Keep temperature low for factual, non-creative responses
                'num_ctx': 4096 # Set context window size
            }
        )
        
        # 4. Extract and clean the response
        # The 'response' key holds the generated text
        llm_text = response.get('response', 'Could not generate AI summary.')
        
        # 5. Store the LLM-enhanced summary in a new key
        guidance['llm_enhanced'] = llm_text.strip()
        
    except Exception as e:
        # This catches errors like Ollama server not running, model not found, or network issues
        guidance['llm_enhanced'] = (
            f"**⚠️ LLM Integration Failed:** Cannot connect to Ollama server or model. "
            f"Error: {e.__class__.__name__}. Please ensure Ollama is running and '{LLM_MODEL_NAME}' is pulled."
        )
    
    return guidance
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
    """Determine the device to use (GPU if available, otherwise CPU)."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def detect_on_image(img_array, model, device):
    """Run detection on a single image array."""
    try:
        # Convert RGB (PIL/Streamlit) to BGR (OpenCV) for display consistency, though YOLO handles RGB
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
        
        # Color coding: Green for no accident, Yellow/Orange/Red for increasing severity/pothole
        if cls_name == 'no_accident':
            color = (0, 255, 0) # Green
        elif 'minor' in cls_name:
            color = (0, 255, 255) # Yellow
        elif 'moderate' in cls_name:
            color = (0, 165, 255) # Orange
        elif 'severe' in cls_name or 'totaled' in cls_name:
            color = (0, 0, 255) # Red
        elif 'pothole' in cls_name:
            color = (255, 255, 0) # Cyan/Blue-ish for structure
        else:
            color = (255, 0, 255) # Magenta for unknown

        label = f"{cls_name} {conf:.2f}"
        
        # Draw bounding box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_bgr, (x1, y1 - 20), (x1 + w, y1), color, -1)
        
        # Draw label text
        cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Convert back to RGB for Streamlit

def detect_on_video(uploaded_file, model, device, detection_placeholder):
    """Process uploaded video file frame by frame."""
    
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    tfile.close() # Crucial fix for PermissionError
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    
    # Initialize results to None. If the loop never runs or the first detection fails,
    # the function will return None, which must be handled by the caller (main()).
    results = None
    
    # Initialize annotated_frame to the original frame in case of early break
    annotated_frame = None

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    st.info(f"Processing video ({width}x{height} @ {fps:.2f} FPS). Please wait...")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Use the last frame for display by default
            frame_to_display = frame.copy()
            
            # Perform detection every 3 frames
            if frame_count % 3 == 0:
                # RUN DETECTION
                yolo_results = model(
                    frame_to_display, 
                    conf=CONF_THRESHOLD, 
                    iou=IOU_THRESHOLD, 
                    device=device,
                    verbose=False
                )
                # Assign to global results variable only if detection was successful
                if yolo_results and len(yolo_results) > 0:
                     results = yolo_results[0]
            
            # DRAW ANNOTATIONS
            # Only draw if a valid results object has been created at least once
            if results is not None:
                # Use the existing results to draw on the current frame
                annotated_frame = draw_detections(frame_to_display, results)
                caption = f"Frame {frame_count} - Detections active: {frame_count % 3 == 0}"
            else:
                # If no detection has run successfully yet, just show the plain frame
                annotated_frame = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
                caption = f"Frame {frame_count} - Waiting for first detection..."
                
            
            # Display the frame in Streamlit
            detection_placeholder.image(
                annotated_frame, 
                channels="RGB", 
                caption=caption, 
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error during video processing: {e}")
        # Ensure 'results' is returned as None if an exception occurs
        results = None
        
    finally:
        # Ensure all resources are released
        cap.release()
        try:
            os.unlink(video_path)
        except PermissionError:
            st.warning("⚠️ Could not immediately delete temporary video file.")
            pass
            
    return results # Return final results object for summary
# ==================== STREAMLIT APP ====================
def main():
    st.set_page_config(
        page_title="Intelligent Road Incident Detection & First Aid System",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title and Introduction
    st.title("🚨 Intelligent Road Incident Detection & First Aid System")
    st.markdown("""
    This application uses a YOLOv8 object detection model to classify road incidents (accidents, potholes) and provides real-time emergency guidance.
    """)
    
    # Load Model and set device
    model = load_model()
    device = get_device()
    if model is None:
        return
    st.sidebar.success(f"Model loaded successfully on {device.upper()}.")

    # --- Sidebar for Settings ---
    st.sidebar.header("Configuration")
    
    global CONF_THRESHOLD, IOU_THRESHOLD 
    CONF_THRESHOLD = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, CONF_THRESHOLD, 0.05)
    IOU_THRESHOLD = st.sidebar.slider("IoU Threshold", 0.0, 1.0, IOU_THRESHOLD, 0.05)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Reporting Options")
    location = st.sidebar.text_input("Incident Location (e.g., 'NH48, near Pune, India')", "")
    notify_authorities = st.sidebar.checkbox("Simulate Emergency Notification", True, 
                                            help="Sends a *simulated* email alert to demo emergency contacts.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Detection Classes")
    st.sidebar.json(CLASS_NAMES)

    # --- Main Content Area ---
    st.header("Upload Media for Analysis")
    upload_type = st.radio("Select Input Type:", ("Image", "Video"), horizontal=True)

    results = None

    if upload_type == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Use columns with equal width for better alignment
            col1, col2 = st.columns([1, 1], gap="large")
            with col1:
                st.subheader("📷 Original Image")
                st.image(image, use_container_width=True)
            with col2:
                with st.spinner("🔍 Analyzing image..."):
                    results = detect_on_image(img_array, model, device)
                    annotated_img = draw_detections(img_array.copy(), results)
                st.subheader("🎯 Detection Results")
                st.image(annotated_img, channels="RGB", use_container_width=True)
            
    elif upload_type == "Video":
        uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])
        
        if uploaded_file is not None:
            st.subheader("🎥 Real-time Analysis")
            detection_placeholder = st.empty()
            
            with st.spinner("Processing video frames..."):
                results = detect_on_video(uploaded_file, model, device, detection_placeholder)

    # --- Post-Detection Summary and Guidance ---
    if results is not None and results.boxes is not None and len(results.boxes) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("📊 Incident Summary")
        
        detections = []
        highest_severity_cls = 'no_accident'
        highest_emergency_level = 0
        
        # Aggregate all detections and find the highest severity
        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            cls_name = CLASS_NAMES.get(cls_id, 'unknown')
            
            # Only consider non-no_accident detections for summary
            if cls_name != 'no_accident':
                detections.append({'class': cls_name, 'confidence': conf})
                
                # Update highest severity
                current_level = ACCIDENT_GUIDANCE.get(cls_name, {'emergency_level': 0})['emergency_level']
                if current_level > highest_emergency_level:
                    highest_emergency_level = current_level
                    highest_severity_cls = cls_name
        
        if not detections and highest_severity_cls == 'no_accident':
            st.success("✅ **Analysis Complete:** No significant road incidents (accidents or potholes) were detected.")
            return

        # Display Summary Metrics (Highest Severity)
        if highest_severity_cls == 'pothole':
            st.warning("🚧 **Pothole Detected** (Highest Incident)")
        elif highest_severity_cls != 'no_accident':
            st.error(f"🚨 **{highest_severity_cls.replace('_', ' ').title()} Detected** (Highest Severity)")
        
        # Display individual detections if multiple types were found
        if len(detections) > 1:
            st.info(f"Found {len(detections)} distinct objects in total. Focusing on **{highest_severity_cls}** for guidance.")

        # --- Guidance Section (Based on highest severity) ---
        if highest_severity_cls != 'no_accident':
            
            st.markdown("---")
            st.subheader("🚑 Emergency Guidance & First Aid Protocol")
            guidance = get_llm_guidance(highest_severity_cls, detections)

            guidance_cols = st.columns(2)
            
            # Column 1: Immediate Actions
            with guidance_cols[0]:
                st.markdown(f"### ⚠️ Immediate Actions ({guidance['severity']})")
                for action in guidance['immediate_actions']:
                    st.markdown(f"**{action}**")

            # Column 2: Emergency Contacts & Warning Signs
            with guidance_cols[1]:
                if highest_severity_cls == 'pothole':
                    contact_type = 'pothole'
                else:
                    contact_type = 'accident'
                    
                st.markdown(f"### 📞 Critical Contacts")
                st.markdown(f"**Department:** {AUTHORITY_CONTACTS[contact_type]['department']}")
                st.markdown(f"**Emergency Line:** {AUTHORITY_CONTACTS[contact_type]['phone']}")
                st.markdown(f"**Email (Demo):** {AUTHORITY_CONTACTS[contact_type]['email']}")
                
                if guidance['warning_signs']:
                    st.markdown("---")
                    st.markdown("### 🔴 Critical Warning Signs (Call 108/911)")
                    for sign in guidance['warning_signs'][:5]: # Show top 5
                        st.markdown(f"- **{sign}**")
                        
            # Full Recommendations Section (Expander)
            with st.expander("📚 Detailed First Aid Recommendations & Legal Tips", expanded=False):
                st.markdown("---")
                st.markdown("### Detailed Recommendations")
                for rec in guidance['recommendations']:
                    st.markdown(f"- {rec}")
                
                st.markdown("---")
                st.markdown("### Actions to AVOID (DO NOT DO)")
                for item in guidance['do_not_do']:
                    st.markdown(f"**- {item}**")

            # LLM Enhanced Summary (If available)
            if 'llm_enhanced' in guidance:
                st.markdown("---")
                st.subheader("🤖 AI-Enhanced Quick Summary")
                st.code(guidance['llm_enhanced'], language='markdown')

            # Authority Notification Trigger
            st.markdown("---")
            if notify_authorities and location:
                if highest_severity_cls == 'pothole':
                    success, message = send_notification_email('pothole', location, None) # Placeholder image
                else:
                    success, message = send_notification_email('accident', location, None) # Placeholder image
                
                if success:
                    st.success(message)
                else:
                    st.warning(message)
            elif notify_authorities and not location:
                 st.warning("⚠️ **Emergency Alert NOT triggered:** Please provide the **Incident Location** in the sidebar to simulate the notification.")
            else:
                 st.info("Emergency notification is currently disabled. Check the 'Simulate Emergency Notification' box in the sidebar to activate.")


if __name__ == "__main__":
    main()
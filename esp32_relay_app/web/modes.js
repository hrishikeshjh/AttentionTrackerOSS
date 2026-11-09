/**
 * CognEn Operating Modes Specification
 * Maps cognitive behavior states to physical 4-channel relay actuations.
 *
 * Supported Modes:
 * 1. proto       - Smart Study Desk / Intelligent Classroom (Lighting, Ventilation, Cooling, Cue)
 * 2. truck       - Fleet & Heavy Machinery Vigilance (Cabin Alarm, AC Blast, Throttle Interlock, Telematics)
 * 3. exam        - High-Stakes Secure Examination Kiosk (Monitor Cutoff, Proctor Alert, Drawer Lock, Spotlight)
 * 4. cleanroom   - Sterile Bio-Lab & Industrial Safety (Dispenser Trip, Laminar Fan, Air Shower, Door Interlock)
 */

export const SYSTEM_MODES = {
  proto: {
    id: "proto",
    name: "Study & Classroom",
    badge: "PROTOTYPE",
    description: "Closed-loop environmental optimization for study desks and smart classrooms.",
    channels: [
      { id: 1, name: "Task Lighting", desc: "Illuminates on eye strain or downward reading", pin: "GPIO 18" },
      { id: 2, name: "Ventilation Exhaust", desc: "Flushes stale CO₂ on drowsiness onset", pin: "GPIO 19" },
      { id: 3, name: "Circulator Fan", desc: "Thermal comfort airflow on restlessness", pin: "GPIO 21" },
      { id: 4, name: "Silent Focus Cue", desc: "Subtle indicator for sustained distraction", pin: "GPIO 22" }
    ],
    evaluate(result) {
      const { behavior, metrics, ols } = result;
      let ch = [0, 0, 0, 0];
      let reason = "Normal operations";

      // Ch 1 (Light): Turn ON if reading book or if eye strain (high blink rate / squint)
      if (behavior === "attentive_reading" || metrics.blinkRate > 24) {
        ch[0] = 1;
        reason = behavior === "attentive_reading" ? "Reading: Task Light Active" : "Eye Strain: Task Light Boosted";
      }

      // Ch 2 (Ventilation): Turn ON if drowsy or OLS fatigue onset anomaly
      if (behavior === "drowsy" || ols.fatigueOnsetAnomaly) {
        ch[1] = 1;
        reason = behavior === "drowsy" ? "Drowsiness: Exhaust Fan Active" : "Fatigue Onset: Purging Stale Air";
      }

      // Ch 3 (Fan): Turn ON if restless fidgeting or slumped posture
      if (metrics.fidget > 55 || metrics.pitchDeg > 22) {
        ch[2] = 1;
        reason = "Restlessness: Circulator Fan Active";
      }

      // Ch 4 (Silent Cue): Pulse / ON if using phone or prolonged distraction
      if (behavior === "using_phone" || behavior === "distracted") {
        ch[3] = 1;
        reason = behavior === "using_phone" ? "Phone Usage: Focus Cue Active" : "Distraction Detected";
      }

      return { channels: ch, reason };
    }
  },

  truck: {
    id: "truck",
    name: "Fleet & Heavy Machinery",
    badge: "SAFETY CRITICAL",
    description: "Vigilance monitoring and emergency interlocks for vehicle and crane operators.",
    channels: [
      { id: 1, name: "Cabin Buzzer & Haptic", desc: "Immediate audio-tactile wake-up alarm", pin: "GPIO 18" },
      { id: 2, name: "AC Cabin Blast", desc: "Rapid cold airflow to counter microsleep", pin: "GPIO 19" },
      { id: 3, name: "Throttle / Hydraulic Cut", desc: "Dead-man machine disable on sustained sleep", pin: "GPIO 21" },
      { id: 4, name: "Hazard Beacon & Fleet SOS", desc: "External safety light and telemetry trigger", pin: "GPIO 22" }
    ],
    evaluate(result) {
      const { behavior, metrics } = result;
      let ch = [0, 0, 0, 0];
      let reason = "Operator Vigilant (System Nominal)";

      // Critical Event: Microsleep or Drowsy
      if (behavior === "drowsy" || metrics.closedDurationMs > 1200) {
        ch[0] = 1; // Buzzer
        ch[1] = 1; // AC blast
        if (metrics.closedDurationMs > 2500) {
          ch[2] = 1; // Throttle interlock (cut power)
          ch[3] = 1; // Beacon SOS
          reason = "EMERGENCY: Prolonged Microsleep! Throttle Interlocked!";
        } else {
          reason = "ALERT: Microsleep Detected! Cabin Wakeup Active";
        }
        return { channels: ch, reason };
      }

      // Distraction / Phone Event
      if (behavior === "using_phone") {
        ch[0] = 1; // Buzzer
        ch[3] = 1; // Warning beacon
        reason = "VIOLATION: In-Cab Mobile Phone Detected!";
        return { channels: ch, reason };
      }

      if (behavior === "distracted") {
        ch[0] = 1; // Gentle buzzer warning
        reason = "WARNING: Operator Gaze Off Road / Off Task";
        return { channels: ch, reason };
      }

      // Absent operator
      if (behavior === "absent") {
        ch[2] = 1; // Interlock machine
        reason = "INTERLOCK: Operator Absent from Seat";
        return { channels: ch, reason };
      }

      return { channels: ch, reason };
    }
  },

  exam: {
    id: "exam",
    name: "Examination & Proctoring",
    badge: "ASSESSMENT",
    description: "Context-aware proctoring hardware interlocks for secure testing terminals.",
    channels: [
      { id: 1, name: "Terminal Monitor Mains", desc: "Instantly blacks out screen on illicit device", pin: "GPIO 18" },
      { id: 2, name: "Proctor Silent Desk Lamp", desc: "Discreetly alerts proctor of violation", pin: "GPIO 19" },
      { id: 3, name: "Solenoid Drawer Lock", desc: "Secures desk storage compartment", pin: "GPIO 21" },
      { id: 4, name: "Seat Status Spotlight", desc: "Overhead compliance spotlight", pin: "GPIO 22" }
    ],
    evaluate(result) {
      const { behavior } = result;
      let ch = [1, 0, 1, 0]; // Default: Monitor ON (1), Drawer LOCKED (1)
      let reason = "Candidate Compliant (Exam Active)";

      // Legitimate reading or scratchpad writing
      if (behavior === "attentive_reading") {
        ch[0] = 1; // Monitor ON
        ch[3] = 1; // Overhead soft light
        reason = "Permitted: Scratchpad Calculation";
        return { channels: ch, reason };
      }

      // Cheating / Illicit phone usage
      if (behavior === "using_phone") {
        ch[0] = 0; // Blackout monitor!
        ch[1] = 1; // Alert proctor desk
        ch[3] = 1; // Spotlight on seat
        reason = "BREACH: Unauthorized Smartphone Detected! Monitor Cut!";
        return { channels: ch, reason };
      }

      // Absent during exam
      if (behavior === "absent") {
        ch[0] = 0; // Blank terminal
        reason = "Candidate Left Seat";
        return { channels: ch, reason };
      }

      if (behavior === "distracted") {
        ch[1] = 1; // Notify proctor of looking around
        reason = "Suspicious: Candidate Looking Away from Terminal";
        return { channels: ch, reason };
      }

      return { channels: ch, reason };
    }
  },

  cleanroom: {
    id: "cleanroom",
    name: "Cleanroom & Bio-Lab",
    badge: "STERILE SAFETY",
    description: "Sterility enforcement and automated equipment cutoff in hazardous laboratories.",
    channels: [
      { id: 1, name: "Dispenser / Robot Cutoff", desc: "Trips pipetting robot on contaminated touch", pin: "GPIO 18" },
      { id: 2, name: "Laminar Flow Booster", desc: "Increases positive air pressure", pin: "GPIO 19" },
      { id: 3, name: "Sterilization Air Shower", desc: "Decontamination purge strobe", pin: "GPIO 21" },
      { id: 4, name: "Airlock Door Interlock", desc: "Secures biohazard containment barrier", pin: "GPIO 22" }
    ],
    evaluate(result) {
      const { behavior } = result;
      let ch = [0, 0, 0, 0];
      let reason = "Sterile Zone Compliant";

      if (behavior === "using_phone") {
        ch[0] = 1; // Trip robot
        ch[2] = 1; // Decontamination strobe
        reason = "CONTAMINATION HAZARD: Non-sterile device in hood!";
        return { channels: ch, reason };
      }

      if (behavior === "drowsy") {
        ch[1] = 1; // Boost airflow
        reason = "Lethargy in Lab: Laminar Air Boosted";
        return { channels: ch, reason };
      }

      return { channels: ch, reason };
    }
  }
};

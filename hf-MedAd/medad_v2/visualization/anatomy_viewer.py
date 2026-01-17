"""
MEDAD 2.0 - 3D Anatomy Viewer
================================

Three.js-compatible 3D human anatomy visualization.
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger("AnatomyViewer")


class AnatomyViewer:
    """
    3D human anatomy visualization for affected area highlighting.
    
    Provides Three.js scene configurations for client-side rendering.
    """
    
    # Body part configurations
    BODY_PARTS = {
        "head": {
            "position": [0, 1.7, 0],
            "scale": [0.12, 0.12, 0.12],
            "default_color": "#FFDBAC",
            "mesh_id": "head_mesh"
        },
        "neck": {
            "position": [0, 1.5, 0],
            "scale": [0.05, 0.1, 0.05],
            "default_color": "#FFDBAC",
            "mesh_id": "neck_mesh"
        },
        "chest": {
            "position": [0, 1.3, 0.05],
            "scale": [0.2, 0.2, 0.1],
            "default_color": "#FFDBAC",
            "mesh_id": "chest_mesh"
        },
        "abdomen": {
            "position": [0, 1.0, 0.05],
            "scale": [0.15, 0.15, 0.1],
            "default_color": "#FFDBAC",
            "mesh_id": "abdomen_mesh"
        },
        "back": {
            "position": [0, 1.2, -0.1],
            "scale": [0.2, 0.3, 0.05],
            "default_color": "#FFDBAC",
            "mesh_id": "back_mesh"
        },
        "left_arm": {
            "position": [-0.25, 1.2, 0],
            "scale": [0.05, 0.3, 0.05],
            "default_color": "#FFDBAC",
            "mesh_id": "left_arm_mesh"
        },
        "right_arm": {
            "position": [0.25, 1.2, 0],
            "scale": [0.05, 0.3, 0.05],
            "default_color": "#FFDBAC",
            "mesh_id": "right_arm_mesh"
        },
        "left_leg": {
            "position": [-0.1, 0.5, 0],
            "scale": [0.07, 0.4, 0.07],
            "default_color": "#FFDBAC",
            "mesh_id": "left_leg_mesh"
        },
        "right_leg": {
            "position": [0.1, 0.5, 0],
            "scale": [0.07, 0.4, 0.07],
            "default_color": "#FFDBAC",
            "mesh_id": "right_leg_mesh"
        },
        "full_body": {
            "position": [0, 1.0, 0],
            "scale": [0.3, 0.8, 0.15],
            "default_color": "#FFDBAC",
            "mesh_id": "body_mesh"
        }
    }
    
    # Symptom to body part mapping
    SYMPTOM_TO_BODY_PART = {
        "headache": ["head"],
        "migraine": ["head"],
        "fever": ["full_body"],
        "cough": ["chest"],
        "cold": ["head", "chest"],
        "sore throat": ["neck"],
        "throat": ["neck"],
        "stomach": ["abdomen"],
        "abdominal": ["abdomen"],
        "back pain": ["back"],
        "arm pain": ["left_arm", "right_arm"],
        "leg pain": ["left_leg", "right_leg"],
        "knee": ["left_leg", "right_leg"],
        "joint": ["left_arm", "right_arm", "left_leg", "right_leg"],
        "skin": ["full_body"],
        "allergy": ["full_body"],
        "fatigue": ["full_body"],
    }
    
    def __init__(self, config=None):
        """Initialize the anatomy viewer"""
        self.config = config
        logger.info("AnatomyViewer initialized")
    
    def get_affected_parts(self, symptom: str) -> List[str]:
        """Determine which body parts are affected by a symptom"""
        symptom_lower = symptom.lower()
        affected = []
        
        for key, parts in self.SYMPTOM_TO_BODY_PART.items():
            if key in symptom_lower:
                affected.extend(parts)
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for part in affected:
            if part not in seen:
                seen.add(part)
                unique.append(part)
        
        return unique if unique else ["full_body"]
    
    def create_scene_config(
        self,
        symptom: str,
        highlight_color: str = "#FF5252",
        animation: bool = True
    ) -> Dict[str, Any]:
        """
        Create a Three.js scene configuration for the symptom.
        
        Returns a JSON-serializable configuration that can be used
        by the frontend Three.js renderer.
        """
        affected_parts = self.get_affected_parts(symptom)
        
        scene_config = {
            "type": "anatomy_scene",
            "version": "1.0",
            "model": {
                "type": "low_poly_human",
                "format": "gltf",
                "url": "/static/models/human_body.glb"
            },
            "camera": {
                "type": "perspective",
                "fov": 45,
                "position": [0, 1.2, 3],
                "target": [0, 1.0, 0],
                "controls": {
                    "enabled": True,
                    "autoRotate": animation,
                    "autoRotateSpeed": 0.5,
                    "enableZoom": True,
                    "minDistance": 1.5,
                    "maxDistance": 5
                }
            },
            "lights": [
                {
                    "type": "ambient",
                    "color": "#ffffff",
                    "intensity": 0.6
                },
                {
                    "type": "directional",
                    "color": "#ffffff",
                    "intensity": 0.8,
                    "position": [5, 10, 5],
                    "castShadow": True
                },
                {
                    "type": "point",
                    "color": highlight_color,
                    "intensity": 0.5,
                    "position": self.BODY_PARTS.get(affected_parts[0], self.BODY_PARTS["full_body"])["position"],
                    "distance": 1
                }
            ],
            "highlights": [],
            "labels": [],
            "animations": []
        }
        
        # Add highlights for affected parts
        for part in affected_parts:
            part_config = self.BODY_PARTS.get(part, self.BODY_PARTS["full_body"])
            
            scene_config["highlights"].append({
                "mesh_id": part_config["mesh_id"],
                "position": part_config["position"],
                "color": highlight_color,
                "opacity": 0.8,
                "pulse": True,
                "pulse_speed": 1.5,
                "glow": True,
                "glow_intensity": 0.5
            })
            
            scene_config["labels"].append({
                "text": symptom.title(),
                "position": [
                    part_config["position"][0] + 0.3,
                    part_config["position"][1],
                    part_config["position"][2]
                ],
                "font_size": 14,
                "color": "#FFFFFF",
                "background": highlight_color,
                "padding": 5
            })
        
        # Add pulse animation
        if animation:
            scene_config["animations"].append({
                "type": "pulse",
                "targets": [h["mesh_id"] for h in scene_config["highlights"]],
                "scale_range": [1.0, 1.1],
                "duration": 1000,
                "repeat": True
            })
        
        return scene_config
    
    def get_organ_info(self, body_part: str) -> Dict[str, Any]:
        """Get detailed information about a body part/organ"""
        organ_info = {
            "head": {
                "name": "Head",
                "contains": ["Brain", "Eyes", "Ears", "Nose", "Mouth"],
                "common_conditions": ["Headache", "Migraine", "Concussion", "Sinusitis"],
                "description": "The head contains the brain and major sensory organs."
            },
            "chest": {
                "name": "Chest/Thorax",
                "contains": ["Heart", "Lungs", "Ribs", "Esophagus"],
                "common_conditions": ["Pneumonia", "Bronchitis", "Heart Disease", "GERD"],
                "description": "The thoracic cavity contains vital respiratory and cardiovascular organs."
            },
            "abdomen": {
                "name": "Abdomen",
                "contains": ["Stomach", "Intestines", "Liver", "Kidneys", "Pancreas"],
                "common_conditions": ["Gastritis", "IBS", "Appendicitis", "Kidney Stones"],
                "description": "The abdominal cavity houses digestive and excretory organs."
            },
            "back": {
                "name": "Back/Spine",
                "contains": ["Vertebral Column", "Spinal Cord", "Back Muscles"],
                "common_conditions": ["Lower Back Pain", "Sciatica", "Herniated Disc"],
                "description": "The back provides structural support and protects the spinal cord."
            }
        }
        
        return organ_info.get(body_part, {
            "name": body_part.replace("_", " ").title(),
            "contains": [],
            "common_conditions": [],
            "description": ""
        })

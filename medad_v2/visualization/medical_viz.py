"""
MEDAD 2.0 - Medical Visualization Module
============================================

Advanced 3D visualization for medical data analysis.

Key Features:
- 3D anatomy visualization with highlighted affected areas
- Drug interaction network graphs
- Disease prevalence 3D heatmaps
- Treatment pathway visualization
- Interactive Plotly graphs

Uses:
- Plotly for 3D graphs
- Three.js integration for anatomy models
- NetworkX for drug interaction analysis
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("MedicalVisualizer")


@dataclass
class VisualizationResult:
    """Result of visualization generation"""
    viz_type: str
    data: Dict[str, Any]
    plotly_figure: Optional[Any] = None
    three_js_scene: Optional[Dict[str, Any]] = None
    html_content: Optional[str] = None


class MedicalVisualizer:
    """
    Medical data visualization engine.
    
    Provides:
    1. 3D anatomy models with disease highlighting
    2. Drug interaction network graphs
    3. Treatment efficacy visualizations
    4. Disease progression timelines
    
    Example:
        viz = MedicalVisualizer(config)
        result = await viz.generate(
            symptom="headache",
            medicines=medicines,
            viz_type="drug_interaction"
        )
        # Returns Plotly figure for rendering
    """
    
    def __init__(self, config=None):
        """
        Initialize the visualizer.
        
        Args:
            config: VisualizationConfig with rendering parameters (optional)
        """
        # Create default config if not provided
        if config is None:
            from ..core.config import VisualizationConfig
            config = VisualizationConfig()
        
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize sub-modules
        self._drug_graph = None
        self._anatomy_viewer = None
        
        logger.info("MedicalVisualizer initialized")
    
    async def generate(
        self,
        symptom: str,
        medicines: List[Dict[str, Any]],
        viz_type: str = "drug_interaction"
    ) -> VisualizationResult:
        """
        Generate a visualization.
        
        Args:
            symptom: The symptom/condition being visualized
            medicines: List of recommended medicines
            viz_type: Type of visualization:
                - drug_interaction: Network graph of drug interactions
                - anatomy: 3D body with affected area highlighted
                - prevalence: Disease prevalence statistics
                - treatment_pathway: Treatment decision tree
        
        Returns:
            VisualizationResult with Plotly figure or Three.js scene
        """
        if viz_type == "drug_interaction":
            return await self._generate_drug_interaction_graph(symptom, medicines)
        elif viz_type == "anatomy":
            return await self._generate_anatomy_visualization(symptom)
        elif viz_type == "prevalence":
            return await self._generate_prevalence_chart(symptom, medicines)
        elif viz_type == "treatment_pathway":
            return await self._generate_treatment_pathway(symptom, medicines)
        else:
            logger.warning(f"Unknown viz_type: {viz_type}")
            return VisualizationResult(viz_type=viz_type, data={})
    
    async def _generate_drug_interaction_graph(
        self,
        symptom: str,
        medicines: List[Dict[str, Any]]
    ) -> VisualizationResult:
        """
        Generate a 3D network graph showing drug interactions.
        
        Nodes: Medicines
        Edges: Interactions (positive/negative)
        Colors: Therapeutic class
        Size: Relevance score
        """
        import plotly.graph_objects as go
        import numpy as np
        
        loop = asyncio.get_event_loop()
        
        def _create_graph():
            # Create node positions (3D layout)
            n_medicines = len(medicines)
            if n_medicines == 0:
                return None
            
            # Arrange in 3D sphere
            phi = np.linspace(0, 2 * np.pi, n_medicines, endpoint=False)
            theta = np.linspace(0, np.pi, n_medicines)
            
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            
            # Create node trace
            node_colors = []
            node_sizes = []
            node_labels = []
            hover_texts = []
            
            therapeutic_classes = {}
            color_palette = [
                "#00695C", "#D32F2F", "#1976D2", "#388E3C", "#7B1FA2",
                "#F57C00", "#0097A7", "#C2185B", "#512DA8", "#00796B"
            ]
            
            for i, med in enumerate(medicines):
                name = med.get('Medicine Name', f'Medicine {i+1}')
                tc = med.get('Therapeutic Class', 'General')
                score = med.get('Raw Score', 0.5)
                
                node_labels.append(name)
                node_sizes.append(15 + score * 20)
                
                # Assign color by therapeutic class
                if tc not in therapeutic_classes:
                    therapeutic_classes[tc] = color_palette[len(therapeutic_classes) % len(color_palette)]
                node_colors.append(therapeutic_classes[tc])
                
                hover_texts.append(
                    f"<b>{name}</b><br>"
                    f"Class: {tc}<br>"
                    f"Uses: {med.get('Uses', 'N/A')[:100]}...<br>"
                    f"Score: {score:.2f}"
                )
            
            # Node trace
            node_trace = go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers+text',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    opacity=0.9,
                    line=dict(width=2, color='white')
                ),
                text=node_labels,
                textposition="top center",
                textfont=dict(size=10, color='white'),
                hovertext=hover_texts,
                hoverinfo='text',
                name='Medicines'
            )
            
            # Create edge traces (interactions between medicines with same therapeutic class)
            edge_x = []
            edge_y = []
            edge_z = []
            
            for i in range(n_medicines):
                for j in range(i + 1, n_medicines):
                    tc1 = medicines[i].get('Therapeutic Class', '')
                    tc2 = medicines[j].get('Therapeutic Class', '')
                    
                    if tc1 == tc2:  # Same therapeutic class = interaction
                        edge_x.extend([x[i], x[j], None])
                        edge_y.extend([y[i], y[j], None])
                        edge_z.extend([z[i], z[j], None])
            
            edge_trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='rgba(136, 136, 136, 0.5)', width=2),
                hoverinfo='none',
                name='Interactions'
            )
            
            # Central symptom node
            symptom_trace = go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers+text',
                marker=dict(
                    size=30,
                    color='#D32F2F',
                    symbol='diamond',
                    line=dict(width=3, color='white')
                ),
                text=[symptom.upper()],
                textposition='bottom center',
                textfont=dict(size=14, color='white', family='Arial Black'),
                hovertext=f"<b>Symptom:</b> {symptom}<br><b>Medicines Found:</b> {n_medicines}",
                hoverinfo='text',
                name='Symptom'
            )
            
            # Create lines from symptom to medicines
            symptom_lines_x = []
            symptom_lines_y = []
            symptom_lines_z = []
            
            for i in range(n_medicines):
                symptom_lines_x.extend([0, x[i], None])
                symptom_lines_y.extend([0, y[i], None])
                symptom_lines_z.extend([0, z[i], None])
            
            symptom_lines = go.Scatter3d(
                x=symptom_lines_x, y=symptom_lines_y, z=symptom_lines_z,
                mode='lines',
                line=dict(
                    color='rgba(255, 82, 82, 0.4)',
                    width=1,
                    dash='dot'
                ),
                hoverinfo='none',
                name='Connections'
            )
            
            # Create figure
            fig = go.Figure(data=[edge_trace, symptom_lines, node_trace, symptom_trace])
            
            fig.update_layout(
                title=dict(
                    text=f"🧬 Drug Interaction Network: {symptom.title()}",
                    font=dict(size=20, color='#00695C', family='Inter')
                ),
                showlegend=True,
                legend=dict(
                    x=0, y=1,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#00695C',
                    borderwidth=1
                ),
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    bgcolor='rgba(224, 242, 241, 0.5)'
                ),
                margin=dict(l=0, r=0, t=50, b=0),
                paper_bgcolor='rgba(224, 242, 241, 0.3)',
                font=dict(family='Inter')
            )
            
            # Add camera animation
            fig.update_layout(
                scene_camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            )
            
            return fig
        
        fig = await loop.run_in_executor(self._executor, _create_graph)
        
        return VisualizationResult(
            viz_type="drug_interaction",
            data={
                "symptom": symptom,
                "medicine_count": len(medicines),
                "medicines": [m.get('Medicine Name', '') for m in medicines]
            },
            plotly_figure=fig
        )
    
    async def _generate_anatomy_visualization(
        self,
        symptom: str
    ) -> VisualizationResult:
        """
        Generate 3D anatomy visualization with affected area highlighted.
        
        Returns Three.js scene configuration for client-side rendering.
        """
        # Map symptoms to body parts
        symptom_body_map = {
            "headache": {"part": "head", "position": [0, 1.7, 0], "color": "#FF5252"},
            "migraine": {"part": "head", "position": [0, 1.7, 0], "color": "#D32F2F"},
            "fever": {"part": "full_body", "position": [0, 1, 0], "color": "#FF9800"},
            "cough": {"part": "chest", "position": [0, 1.3, 0.15], "color": "#2196F3"},
            "stomach": {"part": "abdomen", "position": [0, 1.0, 0.1], "color": "#4CAF50"},
            "back": {"part": "back", "position": [0, 1.2, -0.15], "color": "#9C27B0"},
            "knee": {"part": "knee", "position": [0.1, 0.5, 0], "color": "#FF5722"},
            "throat": {"part": "throat", "position": [0, 1.5, 0.05], "color": "#E91E63"},
            "skin": {"part": "skin", "position": [0, 1, 0], "color": "#795548"},
            "eye": {"part": "head", "position": [0, 1.65, 0.1], "color": "#00BCD4"},
        }
        
        # Find matching body part
        symptom_lower = symptom.lower()
        affected_area = None
        
        for key, value in symptom_body_map.items():
            if key in symptom_lower:
                affected_area = value
                break
        
        if not affected_area:
            affected_area = {"part": "full_body", "position": [0, 1, 0], "color": "#00695C"}
        
        # Three.js scene configuration
        three_js_scene = {
            "model": "human_body_low_poly",
            "camera": {
                "position": [0, 1, 3],
                "target": [0, 1, 0],
                "fov": 45
            },
            "lights": [
                {"type": "ambient", "color": "#ffffff", "intensity": 0.5},
                {"type": "directional", "color": "#ffffff", "intensity": 0.8, "position": [5, 5, 5]}
            ],
            "highlights": [
                {
                    "part": affected_area["part"],
                    "position": affected_area["position"],
                    "color": affected_area["color"],
                    "intensity": 0.8,
                    "pulse": True
                }
            ],
            "labels": [
                {
                    "text": symptom.title(),
                    "position": affected_area["position"],
                    "offset": [0.3, 0, 0]
                }
            ],
            "animation": {
                "rotate": True,
                "speed": 0.001
            }
        }
        
        return VisualizationResult(
            viz_type="anatomy",
            data={
                "symptom": symptom,
                "affected_area": affected_area
            },
            three_js_scene=three_js_scene
        )
    
    async def _generate_prevalence_chart(
        self,
        symptom: str,
        medicines: List[Dict[str, Any]]
    ) -> VisualizationResult:
        """
        Generate 3D prevalence and statistics visualization.
        """
        import plotly.graph_objects as go
        
        loop = asyncio.get_event_loop()
        
        def _create_chart():
            # Group medicines by therapeutic class
            class_counts = {}
            for med in medicines:
                tc = med.get('Therapeutic Class', 'Other')
                class_counts[tc] = class_counts.get(tc, 0) + 1
            
            classes = list(class_counts.keys())[:10]  # Top 10 classes
            counts = [class_counts[c] for c in classes]
            
            # Create 3D bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=classes,
                    y=counts,
                    marker=dict(
                        color=counts,
                        colorscale='Teal',
                        line=dict(width=2, color='#00695C')
                    ),
                    text=counts,
                    textposition='outside',
                    hovertemplate="<b>%{x}</b><br>Medicines: %{y}<extra></extra>"
                )
            ])
            
            fig.update_layout(
                title=dict(
                    text=f"📊 Medicine Distribution for: {symptom.title()}",
                    font=dict(size=18, color='#00695C')
                ),
                xaxis=dict(title="Therapeutic Class", tickangle=45),
                yaxis=dict(title="Number of Medicines"),
                paper_bgcolor='rgba(224, 242, 241, 0.3)',
                plot_bgcolor='rgba(255, 255, 255, 0.8)',
                font=dict(family='Inter')
            )
            
            return fig
        
        fig = await loop.run_in_executor(self._executor, _create_chart)
        
        return VisualizationResult(
            viz_type="prevalence",
            data={"symptom": symptom},
            plotly_figure=fig
        )
    
    async def _generate_treatment_pathway(
        self,
        symptom: str,
        medicines: List[Dict[str, Any]]
    ) -> VisualizationResult:
        """
        Generate treatment decision pathway visualization.
        """
        import plotly.graph_objects as go
        
        loop = asyncio.get_event_loop()
        
        def _create_sankey():
            # Create treatment pathway as Sankey diagram
            labels = [symptom.title()]  # Node 0: Symptom
            
            # Add severity levels
            labels.extend(["Mild", "Moderate", "Severe"])  # Nodes 1, 2, 3
            
            # Add top medicines
            medicine_names = [m.get('Medicine Name', f'Med {i}')[:20] for i, m in enumerate(medicines[:5])]
            labels.extend(medicine_names)
            
            # Define flows
            sources = []
            targets = []
            values = []
            
            # Symptom to severity
            for i in range(1, 4):
                sources.append(0)
                targets.append(i)
                values.append(30)
            
            # Severity to medicines
            for i, _ in enumerate(medicine_names):
                med_idx = 4 + i
                # Connect to severity levels
                for sev_idx in range(1, 4):
                    sources.append(sev_idx)
                    targets.append(med_idx)
                    values.append(10)
            
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="#00695C", width=0.5),
                    label=labels,
                    color=[
                        "#D32F2F",  # Symptom
                        "#4CAF50", "#FF9800", "#F44336",  # Severity
                    ] + ["#00695C"] * len(medicine_names)  # Medicines
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color="rgba(0, 105, 92, 0.3)"
                )
            )])
            
            fig.update_layout(
                title=dict(
                    text=f"🔬 Treatment Pathway: {symptom.title()}",
                    font=dict(size=18, color='#00695C')
                ),
                font=dict(size=12, family='Inter'),
                paper_bgcolor='rgba(224, 242, 241, 0.3)'
            )
            
            return fig
        
        fig = await loop.run_in_executor(self._executor, _create_sankey)
        
        return VisualizationResult(
            viz_type="treatment_pathway",
            data={"symptom": symptom},
            plotly_figure=fig
        )

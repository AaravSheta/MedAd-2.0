"""
MEDAD 2.0 - Drug Interaction Graph
=====================================

NetworkX-based drug interaction analysis and visualization.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger("DrugInteractionGraph")


class DrugInteractionGraph:
    """
    Drug interaction network analysis.
    
    Uses NetworkX to build and analyze drug interaction networks.
    """
    
    # Known drug interaction types
    INTERACTION_TYPES = {
        "synergistic": {"color": "#4CAF50", "weight": 1.0, "description": "Drugs work better together"},
        "antagonistic": {"color": "#F44336", "weight": -1.0, "description": "Drugs reduce each other's effect"},
        "additive": {"color": "#2196F3", "weight": 0.5, "description": "Combined effect equals sum"},
        "potentiating": {"color": "#FF9800", "weight": 0.8, "description": "One drug enhances the other"},
        "contraindicated": {"color": "#D32F2F", "weight": -2.0, "description": "Should not be taken together"},
    }
    
    # Sample interaction database (in production, this would be a full drug interaction DB)
    INTERACTION_DB = {
        ("paracetamol", "ibuprofen"): "additive",
        ("aspirin", "warfarin"): "potentiating",
        ("metformin", "insulin"): "synergistic",
        ("ace_inhibitor", "potassium"): "contraindicated",
        ("antibiotic", "antacid"): "antagonistic",
    }
    
    def __init__(self):
        """Initialize the drug interaction graph"""
        self._graph = None
        self._initialize_graph()
    
    def _initialize_graph(self):
        """Initialize NetworkX graph"""
        try:
            import networkx as nx
            self._graph = nx.Graph()
            logger.info("Drug interaction graph initialized")
        except ImportError:
            logger.warning("NetworkX not available")
            self._graph = None
    
    def add_medicine(self, medicine: Dict[str, Any]):
        """Add a medicine node to the graph"""
        if self._graph is None:
            return
        
        name = medicine.get('Medicine Name', 'Unknown')
        self._graph.add_node(
            name,
            therapeutic_class=medicine.get('Therapeutic Class', 'General'),
            uses=medicine.get('Uses', ''),
            score=medicine.get('Raw Score', 0.5)
        )
    
    def add_interaction(
        self,
        drug1: str,
        drug2: str,
        interaction_type: str = "additive"
    ):
        """Add an interaction edge between two drugs"""
        if self._graph is None:
            return
        
        interaction_info = self.INTERACTION_TYPES.get(interaction_type, self.INTERACTION_TYPES["additive"])
        
        self._graph.add_edge(
            drug1, drug2,
            type=interaction_type,
            weight=interaction_info["weight"],
            color=interaction_info["color"]
        )
    
    def get_interactions(self, drug: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all interactions for a specific drug"""
        if self._graph is None or drug not in self._graph:
            return []
        
        interactions = []
        for neighbor in self._graph.neighbors(drug):
            edge_data = self._graph.edges[drug, neighbor]
            interactions.append((drug, neighbor, edge_data))
        
        return interactions
    
    def find_contraindications(self, medicines: List[str]) -> List[Tuple[str, str]]:
        """Find any contraindicated drug pairs in a list of medicines"""
        contraindications = []
        
        for i, drug1 in enumerate(medicines):
            for drug2 in medicines[i+1:]:
                # Check interaction database
                key = tuple(sorted([drug1.lower(), drug2.lower()]))
                if key in self.INTERACTION_DB:
                    if self.INTERACTION_DB[key] == "contraindicated":
                        contraindications.append((drug1, drug2))
        
        return contraindications
    
    def get_graph_data(self) -> Dict[str, Any]:
        """Get graph data for visualization"""
        if self._graph is None:
            return {"nodes": [], "edges": []}
        
        nodes = []
        for node, data in self._graph.nodes(data=True):
            nodes.append({
                "id": node,
                "label": node,
                **data
            })
        
        edges = []
        for u, v, data in self._graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                **data
            })
        
        return {"nodes": nodes, "edges": edges}

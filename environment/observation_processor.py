import numpy as np
from typing import Dict, Any, List

class ObservationProcessor:
    """
    Transforms raw Kaggle Orbit Wars observations into tokenized tensor representations.
    Normalizes spatial coordinates and scales ships/production values.
    """
    def __init__(self, max_entities: int = 200):
        self.max_entities = max_entities
        # Features: [ID, Owner(4-dim one-hot + 1 for neutral), X, Y, Radius, Ships, Production, Is_Comet, Is_Fleet]
        self.feature_dim = 1 + 5 + 1 + 1 + 1 + 1 + 1 + 1 + 1 # Total 13 features

    def process(self, obs: Dict[str, Any], player_id: int) -> Dict[str, np.ndarray]:
        entities = []
        entity_ids = []
        
        # 1. Process Planets
        for p in obs.get("planets", []):
            # [id, owner, x, y, radius, ships, production]
            feat = self._create_feature_vector(p, is_fleet=0, comet_ids=obs.get('comet_planet_ids', []))
            entities.append(feat)
            entity_ids.append(len(entities))

        # 2. Process Fleets
        for f in obs.get("fleets", []):
            # [id, owner, x, y, angle, from_planet_id, ships]
            # Map fleet data to planet-like structure for the transformer
            fleet_p = [f[0], f[1], f[2], f[3], 0.5, f[6], 0] # Radius 0.5, Production 0
            feat = self._create_feature_vector(fleet_p, is_fleet=1, comet_ids=[])
            entities.append(feat)
            entity_ids.append(len(entities))

        # 3. Padding
        num_entities = len(entities)
        if num_entities < self.max_entities:
            padding_size = self.max_entities - num_entities
            entities.extend([[0.0] * self.feature_dim] * padding_size)
            entity_ids.extend([0] * padding_size)
        else:
            entities = entities[:self.max_entities]
            entity_ids = entity_ids[:self.max_entities]

        return {
            "entities": np.array(entities, dtype=np.float32),
            "entity_ids": np.array(entity_ids, dtype=np.int64),
            "mask": np.array([1] * num_entities + [0] * (self.max_entities - num_entities), dtype=np.float32)
        }

    def _create_feature_vector(self, data: List[Any], is_fleet: int, comet_ids: List[int]) -> List[float]:
        # data: [id, owner, x, y, radius, ships, production]
        id_val, owner, x, y, radius, ships, production = data
        
        # Owner One-Hot: [-1, 0, 1, 2, 3] -> 5 classes
        owner_one_hot = [0.0] * 5
        owner_one_hot[owner + 1] = 1.0
        
        is_comet = 1.0 if id_val in comet_ids else 0.0
        
        return [
            id_val / 500.0,      # ID (normalized)
            *owner_one_hot,      # Owner
            x / 100.0,           # X (normalized)
            y / 100.0,           # Y (normalized)
            radius / 10.0,       # Radius
            ships / 1000.0,      # Ships
            production / 5.0,    # Production
            is_comet,            # Is_Comet
            float(is_fleet)      # Is_Fleet
        ]

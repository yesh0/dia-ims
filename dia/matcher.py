from dataclasses import dataclass

import tqdm
import pyopenms as ms

from dia.config import Config
from dia.featurer import FeatureIntensityMap


@dataclass
class TandemMatcher:
    config: Config
    """The config."""

    def match(self, ms1: FeatureIntensityMap, ms2: FeatureIntensityMap):
        threshold = self.config.require(float, "matcher", "score_threshold")
        matched = {}
        feature: ms.Feature
        for feature in tqdm.tqdm(ms1.feature_map, total=ms1.feature_map.size()):
            matches = [t for t in ms2.match_fragment_features(feature, ms1) if t[0] > threshold]
            if len(matches) > 0:
                matched[feature.getUniqueId()] = matches
        return matched

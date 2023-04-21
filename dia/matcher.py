import typing
from dataclasses import dataclass

import tqdm
import pandas as pd
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
            matches = dict(
                (unique_id, score)
                for score, unique_id in ms2.match_fragment_features(feature, ms1) if score > threshold
            )
            if len(matches) > 0:
                matched[feature.getUniqueId()] = matches
        return matched

    def organize(self, ms1: FeatureIntensityMap, ms2: FeatureIntensityMap, matched: dict[int, dict[int, float]],
                 tab: pd.DataFrame):
        """
        Organizes data into human-processable formats.

        ====
        Data
        ====

        By now, we have several kinds of data stored in different places, which we need to make connections between.

        We are to use some abbreviation to represent the data we know:
        - MS1 & MS2:             MS1 features & MS2 features, each in a different feature map;
        - DECONV & DECONV_SCORE: connections between MS1 / MS2 features and their scores;
        - PEP & PEP_SCORE:       peptide matches against MS1 features and their scores.

        We want to finally compute:
        - Tandem MS spectra for peptides: PEP -> (possibly multiple) MS1 -DECONV-> MS2
          (possibly with coefficients from scores);
        - Protein identification from peptides: PEP -> (possibly multiple) MS1 -DECONV-> MS2
          -> score -identify-> Proteins.
        """
        # Computes matched_rev: ms2 -> ms1, to find out whether there are matches sharing the same secondary ions.
        matched_rev = self._reverse_dict_in_dict(matched)

        # For multiple MS1 features matching the same MS2 feature, we simply use the scores as coefficients.
        def coefficients(scores: dict[int, float]):
            total = sum(scores.values())
            return dict((feature, score / total) for feature, score in scores.items())

        ratios = dict((ms2_feature, coefficients(scores)) for ms2_feature, scores in matched_rev.items())

        # Generates tandem spectra.
        # ms1_id -> (ms1_feature, list[(ratio, ms2_feature)])
        deconvoluted = {}
        for ms1_feature_id, matched_ms2 in matched.items():
            spectrum = []
            for ms2_feature_id in matched_ms2.keys():
                spectrum.append((ratios[ms2_feature_id][ms1_feature_id], ms2.get_feature_by_id(ms2_feature_id)))
            deconvoluted[ms1_feature_id] = (ms1.get_feature_by_id(ms1_feature_id), spectrum)

        peptide_indices = {}
        for _, row in tab.T.items():
            identifier = row["identifier"]
            modifications = row["opt_global_adduct_ion"]
            if identifier not in peptide_indices:
                peptide_indices[identifier] = {}
            if modifications not in peptide_indices[identifier]:
                peptide_indices[identifier][modifications] = []
            peptide_indices[identifier][modifications].append(row)
        proteins = {}
        # This should have been filled by the peptide searcher.
        for ms1_feature, _ in deconvoluted.values():
            for identification in ms1_feature.getPeptideIdentifications():
                for hit in identification.getHits():
                    identifiers = hit.getMetaValue("identifier")
                    sequences = hit.getMetaValue("description")
                    if len(identifiers) != 0 and identifiers[0] != b"null" and len(sequences) != 0:
                        identifier = typing.cast(bytes, identifiers[0]).decode()
                        sequence = typing.cast(bytes, sequences[0]).decode()
                        modifications = hit.getMetaValue("modifications")
                        try:
                            peptides = [peptide for peptide in peptide_indices[identifier][modifications] if
                                        peptide["description"] == sequence]
                        except IndexError:
                            continue
                        if len(peptides) == 0:
                            continue
                        peptide = peptides[0]
                        assert identifier.startswith("Protein:")  # See searcher.py
                        peptide_identifier = identifier[len("Protein:"):]
                        protein_identifier, number = peptide_identifier[:8], int(peptide_identifier[8:])
                        if protein_identifier not in proteins:
                            proteins[protein_identifier] = {}
                        if number not in proteins[protein_identifier]:
                            proteins[protein_identifier][number] = []
                        proteins[protein_identifier][number].append((peptide, ms1_feature.getUniqueId()))
        return deconvoluted, proteins

    @classmethod
    def _reverse_dict_in_dict(cls, dict_in_dict: dict[int, dict[int, float]]):
        reversed_dict = {}
        for i, inner in dict_in_dict.items():
            for j, score in inner.items():
                if j not in reversed_dict:
                    reversed_dict[j] = {}
                reversed_dict[j][i] = score
        return reversed_dict

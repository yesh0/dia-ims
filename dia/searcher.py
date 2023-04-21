from dataclasses import dataclass

import pyopenms as ms
import pyteomics.mztab

from dia import utils
from dia.config import Config
from dia.featurer import FeatureIntensityMap


@dataclass
class ProteinLibrarySearcher:
    config: Config
    """The config."""

    def prepare_library(self, f: str, fasta: str):
        """Reads FASTA entries from the FASTA file."""
        assert self.config.require(str, "peptide_searcher", "library", "modifications") == "IAA"
        assert self.config.require(str, "peptide_searcher", "library", "digestion") == "trypsin"

        mapping_file, struct_mapping = utils.get_cache_files(f, "tsv", "struct.tsv")
        if self.config.require(bool, "peptide_searcher", "library", "cache"):
            if utils.is_cache_recent(f, mapping_file, struct_mapping):
                return mapping_file, struct_mapping

        entries = []
        ms.FASTAFile().load(fasta, entries)
        proteins = [ms.AASequence.fromString(entry.sequence) for entry in entries]
        digested = dict((protein, self.digest(
            self.iaa_modify(
                protein
            ), self.config.require(int, "peptide_searcher", "library", "max_digestion_misses"),
        )) for protein in proteins)
        self.save_digestion(digested, mapping_file, struct_mapping)
        return mapping_file, struct_mapping

    @classmethod
    def save_digestion(cls, digested: dict[ms.AASequence, list[ms.AASequence]], mapping_file: str,
                       struct_mapping_file: str):
        # mapping: chemical formula -> (mass, peptide identifier)
        mapping: dict[str, tuple[float, list[str]]] = {}
        # struct_mapping: identifier -> peptide sequence
        struct_mapping: list[tuple[str, ms.AASequence]] = []
        for protein, fragments in digested.items():
            for fragment in fragments:
                formula: ms.EmpiricalFormula = fragment.getFormula()
                composition = formula.toString()
                identifier = f"Protein:{protein.toString()[:8]}{len(struct_mapping)}"
                if composition in mapping:
                    mapping[composition][1].append(identifier)
                else:
                    mapping[composition] = (formula.getMonoWeight(), [identifier])
                struct_mapping.append((identifier, fragment))
        with open(mapping_file, "w") as m:
            m.write("database_name\tDIGESTED\n")
            m.write("database_version\t1.0\n")
            for formula, (mono_weight, identifiers) in mapping.items():
                labels = "\t".join(identifiers)
                m.write(f"{mono_weight}\t{formula}\t{labels}\n")
        with open(struct_mapping_file, "w") as s:
            for identifier, fragment in struct_mapping:
                s.write(f"{identifier}\t{fragment.toString()}\t?\t?\n")

    @classmethod
    def iaa_modify(cls, seq: ms.AASequence) -> ms.AASequence:
        """Applies carbamidomethyl modification to all Cys amino acid residues."""
        mods = set()
        ms.ModificationsDB().searchModifications(mods, "Carbamidomethyl", "Cys", 0)
        iaa = mods.pop()
        for i, residue in enumerate(seq):
            if residue.getName() == "Cysteine":
                seq.setModification(i, iaa)
        return seq

    @classmethod
    def digest(cls, seq: ms.AASequence, misses=3) -> list[ms.AASequence]:
        """Generates theoretical peptide fragments with Trypsin digestion."""
        dig = ms.ProteaseDigestion()
        frags: list[ms.AASequence] = []
        dig.setMissedCleavages(misses)
        dig.digest(seq, frags)
        frags.sort(key=ms.AASequence.getMonoWeight)

        def eq(f1: ms.AASequence, f2: ms.AASequence):
            return f1.size() == f2.size() and f1.hasSubsequence(f2)

        dedup = [frags[i] for i in range(len(frags)) if (i == 0 or not eq(frags[i], frags[i - 1]))]
        return dedup

    def search(self, f: str, feature_map: FeatureIntensityMap):
        """
        Constructs search library files with a given FASTA file, matches features to possible digested peptides,
        annotates the feature map and returns a data frame with extra info (i.e. scores).
        """
        mapping_file, struct_mapping = self.prepare_library(f, self.config.require(str, "peptide_searcher", "library",
                                                                                   "library_file"))
        cache, annotations = utils.get_cache_files(f, "tab.tsv", "annotated.featureXML")
        if self.config.require(bool, "peptide_searcher", "search", "cache"):
            if utils.is_cache_recent(f, cache, annotations):
                table = ms.MzTab()
                ms.MzTabFile().load(cache, table)
                feature_map.feature_map = ms.FeatureMap()
                ms.FeatureXMLFile().load(annotations, feature_map.feature_map)
                return self.parse_mz_tab(cache)

        ams = ms.AccurateMassSearchEngine()
        ams_params = ams.getParameters()
        ams_params.setValue("mass_error_unit", "Da")
        ams_params.setValue("mass_error_value",
                            self.config.require(float, "peptide_searcher", "search", "mass_error_value"))
        ams_params.setValue("ionization_mode", "positive")
        ams_params.setValue("db:mapping", [mapping_file])
        ams_params.setValue("db:struct", [struct_mapping])
        ams.setParameters(ams_params)

        table = ms.MzTab()
        ams.init()

        ams.run(feature_map.feature_map, table)
        ms.MzTabFile().store(cache, table)
        ms.FeatureXMLFile().store(annotations, feature_map.feature_map)
        return self.parse_mz_tab(cache)

    @classmethod
    def parse_mz_tab(cls, cache):
        return pyteomics.mztab.MzTab(cache)["SML"]

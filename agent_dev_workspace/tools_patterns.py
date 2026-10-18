import os
import tempfile
from typing import List, Literal, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

import PAMI.extras.convert.DF2DB as db_converter
import PAMI.frequentPattern.basic.FPGrowth as fpgrowth_algo
import PAMI.frequentPattern.topk.FAE as fae_algo
import PAMI.frequentPattern.maximal.MaxFPGrowth as maxfpgrowth_algo


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Mines frequent patterns for a specific category using PAMI pattern mining algorithms.

        Builds a fresh category-local DTM with English stop words removed, applies feature
        filtering (variance, tfidf, or term_frequency percentile bands), converts surviving
        features to a transactional database, and mines frequent itemsets.

        Args:
            category_name: The target category name to filter session.dataframe by.
            filtering_method: Vocabulary filtering approach ('variance', 'tfidf', or 'term_frequency').
            algorithm: Mining algorithm to run ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support threshold (count). Required for 'fpgrowth' and 'maxfpgrowth'.
            k: Top-k count. Required for 'topk'.

        Returns:
            A summary dictionary containing mined patterns, pattern count, and execution status.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"status": "error", "error": "No dataframe loaded in session."}

        # Filter dataframe for category
        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        if cat_df.empty:
            return {
                "status": "error",
                "error": f"Category '{category_name}' not found in dataframe.",
            }

        # Validate parameter requirements for algorithms
        if algorithm in ("fpgrowth", "maxfpgrowth"):
            if min_sup is None or min_sup <= 0:
                return {
                    "status": "error",
                    "error": f"Parameter 'min_sup' must be a positive integer for algorithm '{algorithm}'.",
                }
        elif algorithm == "topk":
            if k is None or k <= 0:
                return {
                    "status": "error",
                    "error": "Parameter 'k' must be a positive integer for algorithm 'topk'.",
                }

        # Step 1: Build fresh per-category CountVectorizer
        texts = cat_df["text"].fillna("").tolist()
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm = vectorizer.fit_transform(texts)
        except ValueError:
            # e.g., if all words are stop words or vocabulary is empty
            dtm_dense = np.zeros((len(texts), 0))
            feature_names = np.array([])
        else:
            dtm_dense = dtm.toarray()
            feature_names = np.array(vectorizer.get_feature_names_out())

        if dtm_dense.shape[1] == 0:
            result_id = session.next_result_id("patterns")
            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = []
            summary = {
                "result_id": result_id,
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": "no terms survived stopword removal or vectorization",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Step 2: Calculate scores and select surviving features
        if filtering_method == "variance":
            scores = np.var(dtm_dense, axis=0)
            p_low, p_high = np.percentile(scores, 5), np.percentile(scores, 95)
            surviving_mask = (scores >= p_low) & (scores <= p_high)
        elif filtering_method == "term_frequency":
            scores = np.sum(dtm_dense, axis=0)
            p_low, p_high = np.percentile(scores, 5), np.percentile(scores, 95)
            surviving_mask = (scores >= p_low) & (scores <= p_high)
        elif filtering_method == "tfidf":
            tfidf = TfidfTransformer().fit_transform(dtm_dense).toarray()
            scores = np.mean(tfidf, axis=0)
            p20 = np.percentile(scores, 20)
            surviving_mask = scores >= p20
        else:
            return {
                "status": "error",
                "error": f"Unknown filtering method: '{filtering_method}'",
            }

        surviving_terms = feature_names[surviving_mask]
        surviving_dtm = dtm_dense[:, surviving_mask]

        # Step 3: Handle zero surviving terms
        if len(surviving_terms) == 0:
            result_id = session.next_result_id("patterns")
            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = []
            summary = {
                "result_id": result_id,
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Step 4: Convert to transactional database and run PAMI
        df_surviving = pd.DataFrame(surviving_dtm, columns=surviving_terms)
        
        parsed_patterns = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file_path = os.path.join(tmp_dir, "transactional_db.txt")
            
            # Use PAMI DF2DB conversion matching Master exactly
            converter = db_converter.DF2DB(df_surviving)
            converter.convert2TransactionalDatabase(db_file_path, ">=", 1)

            # Mine with requested algorithm
            if algorithm == "fpgrowth":
                miner = fpgrowth_algo.FPGrowth(db_file_path, minSup=min_sup, sep="\t")
                miner.mine()
                patterns_df = miner.getPatternsAsDataFrame()
            elif algorithm == "topk":
                miner = fae_algo.FAE(db_file_path, k=k, sep="\t")
                miner.mine()
                patterns_df = miner.getPatternsAsDataFrame()
            elif algorithm == "maxfpgrowth":
                miner = maxfpgrowth_algo.MaxFPGrowth(db_file_path, minSup=min_sup, sep="\t")
                miner.mine()
                patterns_df = miner.getPatternsAsDataFrame()

            if patterns_df is not None and not patterns_df.empty:
                # Parse patterns DataFrame
                # Expected columns in PAMI output: Patterns / Pattern, Support
                pattern_col = [c for c in patterns_df.columns if "pattern" in c.lower()][0]
                support_col = [c for c in patterns_df.columns if "support" in c.lower()][0]

                for _, row in patterns_df.iterrows():
                    pat_str = str(row[pattern_col])
                    items = pat_str.strip().split("\t") if "\t" in pat_str else pat_str.strip().split()
                    sup_val = int(row[support_col])
                    parsed_patterns.append({"pattern": items, "support": sup_val})

                # Sort descending by support
                parsed_patterns.sort(key=lambda x: x["support"], reverse=True)

        # Save full pattern list into session.artifacts
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = parsed_patterns

        # Format summary (truncate top 20 for summary display)
        result_id = session.next_result_id("patterns")
        summary = {
            "result_id": result_id,
            "status": "success",
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": len(parsed_patterns),
            "patterns": parsed_patterns[:20],
        }

        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k,
            },
            summary,
        )

        return summary

    return [mine_patterns_tool]

/**
 * Shared application-wide constants for PaperMatcher-AI.
 *
 * INDEXED_PAPER_COUNT  – The verified number of unique papers loaded into the
 *                        live index (TF-IDF matrix + SBERT embeddings).
 *                        Source: len(paper_metadata) at index-build time.
 *
 * DATASET_RAW_COUNT    – The original raw arXiv dataset row count before
 *                        deduplication / filtering (for historical reference only).
 *                        Do NOT use this as the live search corpus size.
 */
export const INDEXED_PAPER_COUNT = 287383;
export const DATASET_RAW_COUNT   = 287421;

/**
 * Returns the correct human-readable score label for a given retrieval_mode.
 *
 * @param {string|null|undefined} retrieval_mode - 'hybrid', 'tfidf', or 'semantic'
 * @returns {string} Display label
 */
export function getScoreLabel(retrieval_mode) {
  switch ((retrieval_mode || '').toLowerCase()) {
    case 'tfidf':
        return 'TF-IDF SIMILARITY SCORE';
    case 'semantic':
        return 'SBERT COSINE SIMILARITY';
    case 'hybrid':
        return 'HYBRID COMBINED SCORE';
      default:
        return 'RETRIEVAL SCORE';
  }
}

/**
 * Returns a short score-type descriptor used in comparison deltas.
 * Only use "cosine" language when both values genuinely are cosine similarities
 * from the same vector space.
 *
 * @param {string|null|undefined} retrieval_mode
 * @returns {string}
 */
export function getScoreDeltaLabel(retrieval_mode) {
  switch ((retrieval_mode || '').toLowerCase()) {
    case 'tfidf':
      return 'lexical-score difference';
    case 'semantic':
      return 'cosine delta';       // semantic scores are genuine SBERT cosines
    case 'hybrid':
        return 'combined-score difference';
      default:
        return 'score difference';
  }
}

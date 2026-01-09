import re, numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

SIMILARITY_THRESHOLD = 0.75
CATEGORY_BOOST = 0.15
TIME_WINDOW_HOURS = 72

def preprocess_article(article: Dict[str, Any]) -> str:
    title = (article.get("title") or "").strip()
    content = (article.get("content") or "").strip()
    text = f"{title}. {content}"
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:2000]

class BiasClustering:
    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = 384
        self.clusters: Dict[int, Dict[str, Any]] = {}
        self.next_cluster_id = 0

    def _compute_centroid(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.shape[0] == 0:
            return np.zeros(self.embedding_dim)
        centroid = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 0 else centroid

    def _is_within_time_window(self, article_time: datetime, cluster_time: datetime) -> bool:
        if not article_time or not cluster_time:
            return True
        time_diff = abs((article_time - cluster_time).total_seconds() / 3600)
        return time_diff <= TIME_WINDOW_HOURS

    def cluster_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_articles = []
        for article in articles:
            text = preprocess_article(article)
            if not text:
                continue

            embedding = self.model.encode([text], convert_to_numpy=True)[0]
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            best_cluster_id = None
            best_similarity = -1.0

            for cluster_id, cluster in self.clusters.items():
                if "last_updated" in cluster:
                    if not self._is_within_time_window(article.get("timestamp"), cluster["last_updated"]):
                        continue
                similarity = float(np.dot(embedding, cluster["centroid"]))

                # Optional: bias-aware boost if article + cluster share category
                if article.get("bias") and any(a.get("bias") == article["bias"] for a in cluster["articles"]):
                    similarity += CATEGORY_BOOST

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster_id = cluster_id

            if best_cluster_id is not None and best_similarity >= self.similarity_threshold:
                cluster = self.clusters[best_cluster_id]
                cluster["articles"].append(article)
                cluster["embeddings"] = np.vstack([cluster["embeddings"], embedding.reshape(1, -1)])
                cluster["centroid"] = self._compute_centroid(cluster["embeddings"])
                cluster["last_updated"] = article.get("timestamp") or datetime.utcnow()
                article["cluster_id"] = best_cluster_id
            else:
                self.clusters[self.next_cluster_id] = {
                    "embeddings": embedding.reshape(1, -1),
                    "articles": [article],
                    "centroid": embedding,
                    "last_updated": article.get("timestamp") or datetime.utcnow()
                }
                article["cluster_id"] = self.next_cluster_id
                self.next_cluster_id += 1

            processed_articles.append(article)
        return processed_articles

    def get_summary(self) -> Dict[int, int]:
        return {cid: len(data["articles"]) for cid, data in self.clusters.items()}

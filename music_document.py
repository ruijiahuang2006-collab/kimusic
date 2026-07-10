from __future__ import annotations
from elasticsearch_dsl import Document, Text, Keyword, Integer, Float, Boolean, Index
from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import Q
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义索引和映射
class Music(Document):
    """音乐文档类，用于Elasticsearch索引"""
    
    # 基本字段
    title = Text(analyzer='standard')
    filename = Keyword()
    url = Keyword()
    mp3_url = Keyword()
    
    # 标签和分类字段
    tags = Keyword(multi=True)
    genre = Keyword(multi=True)
    mood = Keyword(multi=True)
    movement = Keyword(multi=True)
    theme = Keyword(multi=True)
    
    # 音频特征字段
    tempo = Float()
    dynamics_rmse_mean = Float()
    dynamics_rmse_std = Float()
    timbre_mfcc_mean = Float()
    pitch_chroma_mean = Float()
    rhythm_beats_frames = Integer(multi=True)
    error = Keyword(null_value="NULL")
    
    class Index:
        name = "music_therapy"
        settings = {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "text_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"]
                    }
                }
            }
        }

class MusicDocumentManager:
    """音乐文档管理器，用于管理Elasticsearch中的音乐数据"""
    
    def __init__(self, es_host="http://localhost:9200", rebuild_index=False):
        """初始化音乐文档管理器"""
        self.es_host = es_host
        self.index_name = "music_therapy"
        self.music_document=None
        
        # 连接Elasticsearch
        connections.create_connection(hosts=[es_host])
        
        # 检查是否需要重建索引
        if rebuild_index and Index(self.index_name).exists():
            logger.info(f"重建索引模式：删除已存在的索引 {self.index_name}")
            Index(self.index_name).delete()
            # 创建新索引
            Music.init()
            logger.info(f"成功重建索引: {self.index_name}")
        elif Index(self.index_name).exists():
            logger.info(f"索引 {self.index_name} 已存在，跳过创建")
        else:
            # 创建索引
            Music.init()
            logger.info(f"成功创建索引: {self.index_name}")
    
    def create_music_document(self, music_data):
        """从音乐数据创建Music文档对象"""
        try:
            # 创建Music文档实例
            music_doc = Music()
            
            # 设置基本字段
            music_doc.title = music_data.get('title', '')
            music_doc.filename = music_data.get('filename', '')
            music_doc.url = music_data.get('url', '')
            music_doc.mp3_url = music_data.get('mp3_url', '')
            
            # 设置标签和分类字段
            music_doc.tags = [tag.lower() for tag in music_data.get('tags', [])]
            music_doc.genre = [genre.lower() for genre in music_data.get('genre', [])]
            music_doc.mood = [mood.lower() for mood in music_data.get('mood', [])]
            music_doc.movement = [movement.lower() for movement in music_data.get('movement', [])]
            music_doc.theme = [theme.lower() for theme in music_data.get('theme', [])]
            
            # 设置音频特征字段
            audio_features = music_data.get('audio_features', {})
            music_doc.tempo = audio_features.get('tempo', 0.0)
            music_doc.dynamics_rmse_mean = audio_features.get('dynamics_rmse_mean', 0.0)
            music_doc.dynamics_rmse_std = audio_features.get('dynamics_rmse_std', 0.0)
            music_doc.timbre_mfcc_mean = audio_features.get('timbre_mfcc_mean', 0.0)
            music_doc.pitch_chroma_mean = audio_features.get('pitch_chroma_mean', 0.0)
            music_doc.rhythm_beats_frames = audio_features.get('rhythm_beats_frames', [])
            music_doc.error = audio_features.get('error', 'NULL')
            
            self.music_document=music_doc
            return music_doc
            
        except Exception as e:
            logger.error(f"创建音乐文档时出错: {str(e)}")
            self.music_document=None
            return None
    
    def save_music_document(self, music_doc):
        """保存音乐文档到Elasticsearch"""
        try:
            music_doc.save()
            logger.info(f"成功保存音乐文档: {music_doc.title}")
            return True
        except Exception as e:
            logger.error(f"保存音乐文档时出错: {str(e)}")
            return False
    
    def search_music(self, criteria, size=4):
        """基于给定条件搜索音乐"""
        # try:
        # 构建搜索查询
        print("criteria", criteria)
        query = self._build_search_query(criteria)

        print("query", query)
        
        ###record search time
        start_time = time.time()
        # 执行搜索
        search = Music.search()        
        search = search.query(query)
        search = search.extra(size=size)
        
        response = search.execute()
        end_time = time.time()
        search_time = end_time - start_time
        print(f"Search time: {search_time} seconds")
        
        print("response", response)

        # 提取结果
        start_time = time.time()
        tracks = []
        for hit in response:
            track = hit.to_dict()
            print("track", track)
            track['match_score'] = hit.meta.score
            tracks.append(track)
        
        end_time = time.time()
        search_time = end_time - start_time
        print(f"!!!Search time: {search_time} seconds")
        
        logger.info(f"搜索完成，找到 {len(tracks)} 个结果")
        return tracks
            
        # except Exception as e:
        #     logger.error(f"搜索音乐时出错: {str(e)}")
        #     return []
    
    def _build_search_query(self, criteria):
        """构建搜索查询"""
        
        must_queries = []
        should_queries = []
        must_not_queries = []
        
        # 处理tempo偏好
        if criteria.get("tempo_preference"):
            tempo_range = {
                "slow": {"lt": 80},
                "medium": {"gte": 80, "lte": 120},
                "fast": {"gt": 120}
            }.get(criteria["tempo_preference"])
            
            if tempo_range:
                must_queries.append(Q("range", tempo=tempo_range))
        
        # 处理dynamics偏好
        if criteria.get("dynamics_preference"):
            dynamics_range = {
                "soft": {"lt": 0.1},
                "moderate": {"gte": 0.1, "lte": 0.2},
                "intense": {"gt": 0.2}
            }.get(criteria["dynamics_preference"])
            
            if dynamics_range:
                must_queries.append(Q("range", dynamics_rmse_mean=dynamics_range))
        
        # 处理mood关键词
        if criteria.get("mood_keywords"):
            mood_terms = [m.lower() for m in criteria["mood_keywords"]]
            should_queries.append(Q("terms", mood=mood_terms))
            should_queries.append(Q("match", title=" ".join(criteria["mood_keywords"])))
        
        # 处理genre关键词
        if criteria.get("genre_keywords"):
            genre_terms = [g.lower() for g in criteria["genre_keywords"]]
            should_queries.append(Q("terms", genre=genre_terms))
            should_queries.append(Q("match", title=" ".join(criteria["genre_keywords"])))
        
        # 处理需要避免的关键词
        if criteria.get("avoid_keywords"):
            avoid_terms = [a.lower() for a in criteria["avoid_keywords"]]
            must_not_queries.extend([
                Q("terms", tags=avoid_terms),
                Q("terms", mood=avoid_terms),
                Q("terms", genre=avoid_terms),
                Q("terms", theme=avoid_terms)
            ])
        
        # 一步构造布尔查询，避免链式赋值
        bool_params = {}
        if must_queries:
            bool_params['must'] = must_queries
        if should_queries:
            bool_params['should'] = should_queries
            bool_params['minimum_should_match'] = 1
        if must_not_queries:
            bool_params['must_not'] = must_not_queries
        
        bool_query = Q("bool", **bool_params)
        return bool_query
    
    def get_all_music(self):
        """获取所有音乐数据"""
        try:
            search = Music.search()
            search = search.extra(size=10000)  # 获取所有数据
            response = search.execute()
            
            tracks = []
            for hit in response:
                track = hit.to_dict()
                tracks.append(track)
            
            return tracks
            
        except Exception as e:
            logger.error(f"获取所有音乐数据时出错: {str(e)}")
            return []
    
    def get_attribute_options(self, attribute_type, max_items=30):
        """获取指定属性的所有选项"""
        try:
            search = Music.search()
            search = search.extra(size=0)  # 不需要文档，只需要聚合结果
            
            # 添加聚合查询
            search.aggs.bucket('unique_values', 'terms', field=attribute_type, size=max_items)
            
            response = search.execute()
            
            # 提取聚合结果
            buckets = response.aggregations.unique_values.buckets
            options = [bucket.key for bucket in buckets]
            
            return options
            
        except Exception as e:
            logger.error(f"获取属性选项时出错: {str(e)}")
            return []

def create_music_index():
    """创建音乐索引的便捷函数"""
    try:
        manager = MusicDocumentManager()
        logger.info("音乐索引创建成功")
        return manager
    except Exception as e:
        logger.error(f"创建音乐索引失败: {str(e)}")
        return None

if __name__ == "__main__":
    # 测试创建索引
    manager = create_music_index()
    if manager:
        print("音乐索引创建成功！")
    else:
        print("音乐索引创建失败！") 
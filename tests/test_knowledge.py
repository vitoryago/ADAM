"""Smoke tests for knowledge layer."""


def test_dbt_knowledge_importable():
    from adam.knowledge.dbt_knowledge import DBTKnowledgeService
    assert DBTKnowledgeService is not None


def test_sql_knowledge_importable():
    from adam.knowledge.sql_knowledge import SQLKnowledgeService
    assert SQLKnowledgeService is not None


def test_dbt_context_detection():
    from adam.knowledge.dbt_knowledge import DBTKnowledgeService
    service = DBTKnowledgeService()
    assert service.detect_dbt_context("create a dbt incremental model") == True
    assert service.detect_dbt_context("what is the weather today") == False


def test_dbt_analyzer_importable():
    from adam.knowledge.dbt_analyzer.parser import DBTManifestParser
    assert DBTManifestParser is not None


def test_knowledge_package_exports():
    from adam.knowledge import DBTKnowledgeService, SQLKnowledgeService
    assert DBTKnowledgeService is not None
    assert SQLKnowledgeService is not None


def test_lineage_service_importable():
    from adam.knowledge.lineage_service import LineageService, LineageAnalyzer
    assert LineageService is not None
    assert LineageAnalyzer is not None


def test_dbt_knowledge_model_layer():
    from adam.knowledge.dbt_knowledge import DBTKnowledgeService, ModelLayer
    service = DBTKnowledgeService()
    assert service.identify_model_layer("staging table for users") == ModelLayer.STAGING
    assert service.identify_model_layer("fact orders table") == ModelLayer.MART_FACT


def test_sql_knowledge_detect_context():
    from adam.knowledge.sql_knowledge import SQLKnowledgeService
    service = SQLKnowledgeService()
    assert service.detect_sql_context("optimize this SQL query") == True
    assert service.detect_sql_context("what color is the sky") == False

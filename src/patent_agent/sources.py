"""Explicit source identities keep similarly numbered treaties separate."""

from typing import Literal, get_args

SourceType = Literal[
    "patent_law",
    "implementing_rules",
    "examination_guidelines",
    "paris_convention",
    "trips",
    "pct",
    "pct_regulations",
    "pct_instructions",
    "budapest_treaty",
    "budapest_regulations",
    "strasbourg_agreement",
    "locarno_agreement",
    "hague_geneva_act",
    "hague_stockholm_act",
    "hague_protocol",
    "hague_act",
    "hague_regulations",
    "hague_instructions",
]
SOURCES = get_args(SourceType)
DOMESTIC_SOURCES = SOURCES[:3]

SOURCE_GUIDE = """国际资料 source_type：paris_convention 巴黎公约；trips TRIPS协定；pct 专利合作条约；
pct_regulations PCT实施细则；pct_instructions PCT行政规程；budapest_treaty 布达佩斯条约；
budapest_regulations 布达佩斯实施细则；strasbourg_agreement 斯特拉斯堡协定；locarno_agreement 洛迦诺协定；
hague_geneva_act 海牙1999日内瓦文本；hague_act 海牙1960文本；hague_stockholm_act 斯德哥尔摩补充文本；
hague_protocol 海牙1960议定书（所载文本注记尚未生效）；hague_regulations 海牙实施细则；hague_instructions 海牙行政规程。
条约正文用article:4bis（第四条之二）；细则用rule:13bis；行政规程用section:101。
不同条约、文本和修订版本不能混同；条约义务不当然等于每一国家的直接适用规则，中国缔约状态和声明须有单独依据。
英文原文可用中文解释，但直接引文保留英文；中文解释不冒充官方译文。文件版本、语言、生效信息及覆盖警告随证据提供。
"""

CREATE DATABASE IF NOT EXISTS LEGAL_DOCS_DB;

CREATE OR REPLACE WAREHOUSE LEGAL_DOCS_DB_wh WITH
     WAREHOUSE_SIZE='X-SMALL'
     AUTO_SUSPEND = 120
     AUTO_RESUME = TRUE
     INITIALLY_SUSPENDED=TRUE;

 USE WAREHOUSE LEGAL_DOCS_DB_wh;

 CREATE OR REPLACE STAGE LEGAL_DOCS_DB.public.fomc
    DIRECTORY = (ENABLE = TRUE)
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

 CREATE OR REPLACE FUNCTION LEGAL_DOCS_DB.public.text_chunker(file_url STRING)
    RETURNS TABLE (chunk VARCHAR)
    LANGUAGE PYTHON
    RUNTIME_VERSION = '3.10'
    HANDLER = 'text_chunker'
    PACKAGES = ('snowflake-snowpark-python', 'PyPDF2','langchain')
    AS
$$
from snowflake.snowpark.types import StringType, StructField, StructType
from langchain.text_splitter import  TextSplitter
from snowflake.snowpark.files import SnowflakeFile
import PyPDF2, io
import re
import logging
import pandas as pd

class CustomRegexTextSplitter(TextSplitter):
    def __init__(self, pattern, chunk_size, chunk_overlap, length_function=len):
        self.pattern = pattern
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def split_text(self, text):
        splits = [match.start() for match in re.finditer(self.pattern, text)]
        splits.append(len(text)) 

        chunks = []
        start = 0

        for end in splits:
            if start < end:
                chunk = text[start:end]
                if self.length_function(chunk) <= self.chunk_size:
                    chunks.append(chunk)
                start = max(end - self.chunk_overlap, start)
        return chunks


class text_chunker:
    def read_txt(self, file_url: str) -> str:
        logger = logging.getLogger("udf_logger")
        logger.info(f"Opening file {file_url}")

        with SnowflakeFile.open(file_url, 'rb') as f:
            buffer = io.BytesIO(f.readall())

        reader = PyPDF2.PdfReader(buffer)
        text = ""
        for page in reader.pages:
            try:
                text += page.extract_text().replace('\0', ' ')
            except:
                text = "Unable to Extract"
                logger.warning(f"Unable to extract from file {file_url}, page {page}")

        return text


    def process(self, file_url: str):
        text = self.read_txt(file_url)

        text_splitter = CustomRegexTextSplitter(
            pattern=r"(\nSection|\n\nSection)",
            chunk_size=8000,
            chunk_overlap=0,
            length_function=len
        )

        chunks = text_splitter.split_text(text)
        df = pd.DataFrame(chunks, columns=['chunk'])

        yield from df.itertuples(index=False, name=None)
$$;


LEGAL_DOCS_DB.PUBLIC.FOMCLEGAL_DOCS_DB.PUBLIC.FOMC
CREATE OR REPLACE TABLE LEGAL_DOCS_DB.public.docs_chunks_table AS
    SELECT
        relative_path,
        build_scoped_file_url(@LEGAL_DOCS_DB.public.fomc, relative_path) AS file_url,
        -- preserve file title information by concatenating relative_path with the chunk
        CONCAT(relative_path, ': ', func.chunk) AS chunk,
        'English' AS language
    FROM
        directory(@LEGAL_DOCS_DB.public.fomc),
        TABLE(LEGAL_DOCS_DB.public.text_chunker(build_scoped_file_url(@LEGAL_DOCS_DB.public.fomc, relative_path))) AS func;



SELECT *
FROM LEGAL_DOCS_DB.public.docs_chunks_table
LIMIT 5;



CREATE OR REPLACE CORTEX SEARCH SERVICE LEGAL_DOCS_DB.public.fomc_meeting
    ON chunk
    ATTRIBUTES language
    WAREHOUSE = LEGAL_DOCS_DB_wh
    TARGET_LAG = '1 hour'
    AS (
    SELECT
        chunk,
        relative_path,
        file_url,
        language
    FROM LEGAL_DOCS_DB.public.docs_chunks_table
    );



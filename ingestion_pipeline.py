import json, shutil, re, constants, glob

from typing import Any, Dict, Tuple
from typing import List
from pathlib import Path

# Unstructured for document parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# LangChain components
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage


def partition_document(file_path: str):
    """Extract elements from PDF using unstructured"""
    print(f"📄 Partitioning document: {file_path}")

    elements = partition_pdf(
        filename=file_path,  # Path to your PDF file
        strategy="hi_res",  # Use the most accurate (but slower) processing method of extraction
        infer_table_structure=True,  # Keep tables as structured HTML, not jumbled text
        extract_image_block_types=["Image"],  # Grab images found in the PDF
        extract_image_block_to_payload=True  # Store images as base64 data you can actually use
    )

    print(f"✅ Extracted {len(elements)} elements")
    return elements


def create_chunks_by_title(element):
    """Create intelligent chunks using title-based strategy"""
    print("🔨 Creating smart chunks...")

    chunks = chunk_by_title(
        element,  # The parsed PDF elements from previous step
        max_characters=3000,  # Hard limit - never exceed 3000 characters per chunk
        new_after_n_chars=2400,  # Try to start a new chunk after 2400 characters
        combine_text_under_n_chars=800  # Merge tiny chunks under 500 chars with neighbors
    )

    # TODO: TO DELETE (THIS IS FOR TESTING ONLY)
    # Keep only the first max_chunks
    # chunks = chunks[:10]

    # store sample chunks in file.
    with open("monitor/sample_chunks.txt", "a", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks[:15]):
            text = str(chunk)
            f.write(f"--- Chunk {i + 1} ({len(text)} chars) ---\n")
            f.write(text + "\n\n")

    print(f"✅ Created {len(chunks)} chunks")
    return chunks


def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    
    # Check for tables and images in original elements
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            
            # Handle tables
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            
            # Handle images
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    
    content_data['types'] = list(set(content_data['types']))
    return content_data


def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content"""
    
    try:
        # Initialize LLM (needs vision model for images)
        llm = constants.chat_model
        
        # Build the text prompt
        prompt_text = f"""You are analyzing visual elements extracted from a PDF, for a RAG pipeline in the field of cybersecurity.

        STEP 1: Decide if it's RELEVANT or IRRELEVANT.
        IRRELEVANT = logos, watermarks, decorative backgrounds, icons, banners, links, headers and footers, 
        stock photos, signatures, QR codes, blank/corrupted content, or repeated 
        boilerplate.
        RELEVANT = tables, charts, diagrams, schematics, or photos that convey 
        actual information relevant to cybersecurity.

        STEP 2: If RELEVANT, Generate a comprehensive, searchable description of the tables and/or images. Make it detailed and searchable - prioritize findability over brevity.

        Write them under \"Table Description (number)\" and \"Image Description (number)\". Do not write anything after.

        - Never write phrases like "no tables are present" or "the image appears to be blank" — if there's nothing valuable, simply produce no output for that item.
        - If something belongs to IRRELEVANT, do not even create a description for it.
        - If NONE of the tables or images have valuable content, output nothing at all — return an empty response.
        - Do not write anything after the descriptions.
        

        """
        
        # Add tables if present
        if tables:
            prompt_text += "CONTENT TO ANALYZE:\n\nTABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"

        # if images:
        #     prompt_text += f"IMAGES:\n{len(images)} image(s) attached below.\n\n"

        # prompt_text += """
        # YOUR TASK:
        # Generate a comprehensive, searchable description of the tables and/or images that covers:

        # 1. Key facts, numbers, and data points from text and tables
        # 2. Main topics and concepts discussed  
        # 3. Questions this content could answer
        # 4. Visual content analysis (charts, diagrams, patterns in images)
        # 5. Alternative search terms users might use

        # Make it detailed and searchable - prioritize findability over brevity.

        # Write them under \"Table Description (number)\" and \"Image Description (number)\". Do not write anything after.
        
        # IMPORTANT — SKIPPING RULES:
        # - If a table has no meaningful data (e.g. empty, decorative, or purely structural with no real values), DO NOT write a "Table Description" for it at all. Skip it completely — no header, no explanation, no mention that it was skipped.
        # - If an image has no meaningful visual content (e.g. blank, a logo, a watermark, a decorative header/footer graphic), DO NOT write an "Image Description" for it at all. Skip it completely — no header, no explanation, no mention that it was skipped.
        # - Never write phrases like "no tables are present" or "the image appears to be blank" — if there's nothing valuable, simply produce no output for that item.
        # - If NONE of the tables or images have valuable content, output nothing at all — return an empty response.

        # Do not write anything after the descriptions."""

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add images to the message
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])

        # Write the raw AI summary to a file.
        with open("monitor/ai_enhanced_summaries.txt", "a", encoding="utf-8") as f:
            f.write(response.content)
            f.write("\n\n" + "=" * 80 + "\n\n")
        
        return response.content
        
    except Exception as e:
        print(f"     ❌ AI summary failed: {e}")
        # Fallback to simple summary
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary


def extract_descriptions(response_text: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Extract 'Table Description (N)' and 'Image Description (N)' sections
    from an LLM response.

    Returns:
        (table_descriptions, image_descriptions) - each a dict mapping
        1-based index -> description text.
    """

    # Matches a label like "Table Description (1)" or "Image Description (2)",
    # capturing the label type, its number, and everything until the next
    # label of either type or end of string.
    pattern = re.compile(
        r"(Table|Image) Description \((\d+)\)\s*\n(.*?)"
        r"(?=(?:Table|Image) Description \(\d+\)|\Z)",
        re.DOTALL
    )

    table_descriptions: Dict[int, str] = {}
    image_descriptions: Dict[int, str] = {}

    for label, number, body in pattern.findall(response_text):
        cleaned = body.strip()
        idx = int(number)
        if label == "Table":
            table_descriptions[idx] = cleaned
        else:
            image_descriptions[idx] = cleaned

    # Write results to file before returning
    with open("monitor/tables_and_images_descriptions.txt", "a", encoding="utf-8") as f:
        for idx in sorted(table_descriptions):
            f.write(f"Table Description ({idx})\n{table_descriptions[idx]}\n\n")
        for idx in sorted(image_descriptions):
            f.write(f"Image Description ({idx})\n{image_descriptions[idx]}\n\n")

    return table_descriptions, image_descriptions


def append_summaries_to_text(data: dict[str, Any], tables: Dict[int, str], images: Dict[int, str]):
    if tables:
        data['text'] += "\n\nTable(s) description(s):\n"
        for idx in sorted(tables):
            data['text'] += f"{tables[idx]}\n"
    
    if images:
        data['text'] += "\n\nImage(s) description(s):\n"
        for idx in sorted(images):
            data['text'] += f"{images[idx]}\n"

    return data


def summarise_chunks(chunks):
    """Process all chunks with AI Summaries"""
    print("🧠 Processing chunks with AI Summaries...")
    
    langchain_documents = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f"   Processing chunk {current_chunk}/{total_chunks}")
        
        # Analyze chunk content
        content_data = separate_content_types(chunk)
        
        # Debug prints
        print(f"     Types found: {content_data['types']}")
        print(f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")
        
        # Create AI-enhanced summary if chunk has tables/images
        if content_data['tables'] or content_data['images']:
            print(f"     → Creating AI summary for mixed content...")
            try:
                extended_content = create_ai_enhanced_summary(
                    content_data['text'],
                    content_data['tables'], 
                    content_data['images']
                )
                print(f"     → AI summary created successfully")
                print(f"     → Enhanced content preview: {extended_content[:200]}...")
            except Exception as e:
                print(f"     ❌ AI summary failed: {e}")
                extended_content = content_data['text']
        else:
            print(f"     → Using raw text (no tables/images)")
            extended_content = content_data['text']
        
        # Create LangChain Document with rich metadata
        # doc = Document(
        #     page_content=enhanced_content,
        #     metadata={
        #         "original_content": json.dumps({
        #             "raw_text": content_data['text'],
        #             "tables_html": content_data['tables'],
        #             "images_base64": content_data['images']
        #         })
        #     }
        # )

        # extract tables and images descriptions.
        tables_dict, images_dict = extract_descriptions(extended_content)

        # Append descriptions to content_data['text']
        content_data = append_summaries_to_text(content_data, tables_dict, images_dict)

        # Create LangChain Document
        doc = Document(
            page_content=content_data['text'],
            metadata={
                "original_content": json.dumps({
                    "tables_html": content_data['tables'],
                    "images_base64": content_data['images']
                })
            }
        )
        
        langchain_documents.append(doc)

    # Store sample langchain docs.
    with open("monitor/sample_langchain_docs.txt", "a", encoding="utf-8") as f:
        for i, doc in enumerate(langchain_documents[:15]):
            f.write(f"--- Doc {i + 1} ---\n")
            f.write(f"page_content ({len(doc.page_content)} chars):\n{doc.page_content}\n\n")
            f.write(f"metadata:\n{doc.metadata}\n\n")
            f.write("=" * 80 + "\n\n")

    print(f"✅ Processed {len(langchain_documents)} chunks")
    return langchain_documents


def create_vector_store(documents, persist_directory):
    """Create and persist ChromaDB vector store"""
    print("🔮 Creating embeddings and storing in ChromaDB...")

    # Delete db if exists.
    if Path(persist_directory).exists():
        shutil.rmtree(persist_directory)
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=constants.embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"✅ Vector store created and saved to {persist_directory}")
    return vectorstore

def empty_monitor_files():
    for filepath in glob.glob("monitor/*.txt"):
        open(filepath, "w").close()


if __name__ == '__main__':

    """Run the complete RAG ingestion pipeline for all PDFs in ./docs."""

    print("🚀 Starting RAG Ingestion Pipeline")
    print("=" * 50)

    empty_monitor_files()

    # Find all PDF documents
    pdf_files = sorted(constants.docs_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"❌ No PDF files found in {constants.docs_dir}")
        exit(1)

    print(f"📚 Found {len(pdf_files)} PDF document(s)")

    all_chunks = []

    # Process every document
    for i, pdf_path in enumerate(pdf_files, start=1):

        print("\n" + "-" * 50)
        print(f"📄 Document {i}/{len(pdf_files)}: {pdf_path.name}")
        print("-" * 50)

        # Step 1: Partition
        print("🔹 Step 1: Partitioning document...")
        elements = partition_document(str(pdf_path))

        # Step 2: Chunk
        print("🔹 Step 2: Creating chunks...")
        chunks = create_chunks_by_title(elements)

        # Add source information to every chunk
        # for chunk in chunks:
        #     chunk.metadata.filename = pdf_path.name

        # Step 3: AI Summarisation
        print("🔹 Step 3: Summarising chunks...")
        document_chunks = summarise_chunks(chunks)

        # Add to global collection
        all_chunks.extend(document_chunks)

        print(
            f"✅ Finished {pdf_path.name}: "
            f"{len(chunks)} chunks → "
            f"{len(document_chunks)} summarised chunks"
        )

    # Step 4: Create ONE vector store containing all documents
    print("\n" + "=" * 50)
    print("🔹 Step 4: Creating Chroma vector store...")
    print(f"📦 Total chunks: {len(all_chunks)}")

    db = create_vector_store(
        all_chunks,
        persist_directory=constants.persist_directory
    )

    print("\n" + "=" * 50)
    print("🎉 RAG ingestion pipeline completed successfully!")
    print(f"📚 Documents processed: {len(pdf_files)}")
    print(f"🧩 Total chunks indexed: {len(all_chunks)}")
    print(f"💾 Vector store: {constants.persist_directory}")
    print("=" * 50)
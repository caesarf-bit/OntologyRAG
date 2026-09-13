import json, constants
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

def export_chunks_to_json(chunks, filename):
    """Export processed chunks to clean JSON format"""
    export_data = []
    
    for i, doc in enumerate(chunks):
        chunk_data = {
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content,
            "metadata": {
                "original_content": json.loads(doc.metadata.get("original_content", "{}"))
            }
        }
        export_data.append(chunk_data)
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported {len(export_data)} chunks to {filename}")
    return export_data


def generate_final_answer(chunks, query):
    """Generate final answer using multimodal content"""
    
    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatOpenAI(model="openai.gpt-5.4-nano", temperature=0)
        
        # Build the text prompt
        prompt_text = f"""Based on the following documents, please answer this question: {query}

    CONTENT TO ANALYZE:
    """
        
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i+1} ---\n"

            # Add raw text
            raw_text = chunk.page_content
            if raw_text:
                prompt_text += f"TEXT:\n{raw_text}\n\n"
            
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                
                # Add raw text
                # raw_text = original_data.get("raw_text", "")
                # if raw_text:
                #     prompt_text += f"TEXT:\n{raw_text}\n\n"
                
                # Add tables as HTML
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"
            
            prompt_text += "\n"
        
        prompt_text += """
    Use the context above to answer the question directly and naturally. Do not mention "the context," "the provided material," "the document," or similar phrases — just answer the question. If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."""

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add all images from all chunks
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                images_base64 = original_data.get("images_base64", [])
            
                for image_base64 in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        with open("monitor/prompt.txt", "a", encoding="utf-8") as f:
                    f.write(str(message))
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."


if __name__ == '__main__':

    db = Chroma(
        persist_directory=constants.persist_directory,
        embedding_function=constants.embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    query = "How to prevent Data Poisoning Attack?"
    retriever = db.as_retriever(search_kwargs={"k": 3})
    chunks = retriever.invoke(query)

    # Export to JSON
    export_chunks_to_json(chunks, "monitor/rag_results.json")

    final_answer = generate_final_answer(chunks, query)
    print(final_answer)
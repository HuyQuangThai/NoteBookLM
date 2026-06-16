from services.chroma import chroma_service

result = chroma_service.get_all_chunks(doc_id = "161c681c-c65f-405f-8517-986b99bc8554")
print(result)
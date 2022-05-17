""" RoBERTa. """


def create_instances_from_document(all_documents, document_index, max_seq_length):
    document = all_documents[document_index]
    instances = []

    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.extend(segment)
        current_length += len(segment)
        i += 1
        if current_length >= max_seq_length:
            instances.append([current_chunk])
            current_chunk = []
            current_length = 0
    if current_chunk:
        instances.append([current_chunk])

    return instances

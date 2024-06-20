def get_most_similar_section(query, sections):
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    
    tfidf_sections = vectorizer.fit_transform(sections)
    
    tfidf_query = vectorizer.transform([query])
    
    similarities = cosine_similarity(tfidf_query, tfidf_sections).flatten()
    
    most_similar_ind = np.argmax(similarities)
    
    return most_similar_ind
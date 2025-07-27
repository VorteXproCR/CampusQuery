    # -------- Chat Input --------
    # cols = st.columns([1, 7])
    # with cols[0]:
    #     target_language = st.selectbox("🌐", ["English", "Hindi", "Spanish"], label_visibility="collapsed")
    # with cols[1]:
    #     # st.markdown("Type your question below 👇")
    #     prompt = st.chat_input("Ask something...")
    
    
 # if prompt:
    #     paper_link = get_question_paper_link(prompt)
    #     if paper_link:
    #         response = f"Here is the question paper link you asked for:\n\n[Click to View Paper]({paper_link})"
    #         st.chat_message("assistant").markdown(response)
    #         st.session_state.messages.append({
    #             "role": "assistant",
    #             "content": response
    #         })
    #         return
    #     # source_lang_code = LANGUAGE_CODES.get(target_language, "en")
    #     src_lang = LANGUAGES[selected_language]
    #     tgt_lang = "en_XX"

    #     st.chat_message("user").markdown(prompt)
    #     st.session_state.messages.append({"role": "user", "content": prompt})
        


    #     # Extract metadata from the prompt
    #     # filter_meta = extract_metadata_from_prompt(prompt)
    #     filter_meta = get_question_paper_link(prompt)

    #     # Initialize retriever with or without filter
    #     if all(filter_meta.values()):
    #         retriever = vectorstore.as_retriever(
    #             search_kwargs={
    #                 "k": 5,
    #                 "filter": filter_meta
    #             }
    #         )
    #     else:
    #         retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    #     # Check if any documents match the query
    #     retrieved_docs = retriever.get_relevant_documents(prompt)

    #     # Filter out empty or meaningless docs
    #     meaningful_docs = [doc for doc in retrieved_docs if doc.page_content.strip() != ""]

    #     if len(meaningful_docs) == 0:
    #         fallback_response = "Sorry, I couldn't find anything relevant in the documents you uploaded to answer that."
    #         st.chat_message("assistant").markdown(fallback_response)
    #         st.session_state.messages.append({
    #             "role": "assistant",
    #             "content": fallback_response
    #         })
    #     else:
    #         from langchain.chains.question_answering import load_qa_chain

    #         chain = load_qa_chain(llm, chain_type="stuff", prompt=custom_prompt())
    #         # chain = load_qa_chain(llm, chain_type="stuff", prompt=custom_prompt(target_language))
    #         # result = chain.run(input_documents=meaningful_docs, question=prompt)
    #         result = chain.run(
    #             input_documents=meaningful_docs,
    #             question = prompt,
    #             # question=prompt,
    #             # selected_language=target_language
    #         )

    #         translated_response = safe_translate(result, "en_XX", LANGUAGES[selected_language], tokenizer, translation_model)
            
    #         # final_response = clean_response(result)
    #         final_response = translated_response if len(translated_response.split()) > 5 else result
    #         final_response = clean_response(final_response)
            
    #         # Check if any relevant image exists for the algorithm or topic in the prompt
    #         image_markdown = get_image_from_prompt(prompt)  # This will return the markdown for the image
    #         if image_markdown:
    #             final_response += f"\n\n{image_markdown}"


    #         st.chat_message("assistant").markdown(final_response)
    #         st.session_state.messages.append({
    #             "role": "assistant",
    #             "content": final_response
    #         })
    
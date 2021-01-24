def generate_examples(filepath):
    assert key in ['train', 'testa', 'testb']
    logging.info("⏳ Generating examples from = %s", filepath)
    with open(filepath, encoding="utf-8") as f:
        guid = 0
        doc_name = ""
        tokens = []
        ner_tags = []
        entity_ids = []
        entity_names = []

        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if tokens:
                    assert doc_name != ""
                    if key != 'train' and key not in doc_name:
                        pass
                    else:
                        '''
                        yield guid, {
                            "id": str(guid),
                            "doc_name": doc_name,
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                            "entity_ids": entity_ids,
                            "entity_names": entity_names,
                        }
                        '''
                    guid += 1
                    # doc_name does not change
                    tokens = []
                    ner_tags = []
                    entity_ids = []
                    entity_names = []

                # **YD** append elements before changing the doc_name
                if line.startswith("-DOCSTART-"):
                    assert line.startswith("-DOCSTART- (")
                    assert line.endswith(")\n")
                    doc_name = line[len("-DOCSTART- ("): -2]

            else:
                # EL_aida tokens are tab separated
                # len(splits) = [1, 4, 6, 7]
                # 1: single symbol
                # 4: ['Tim', 'B', "Tim O'Gorman", '--NME--']
                # 6: ['House', 'B', 'House of Commons', 'House_of_Commons', 'http://en.wikipedia.org/wiki/House_of_Commons', '216091']
                # 7: ['German', 'B', 'German', 'Germany', 'http://en.wikipedia.org/wiki/Germany', '11867', '/m/0345h']

                splits = line.rstrip().split("\t")
                assert len(splits) in [1, 4, 6, 7]

                # **YD** gets out of unicode storing in the entity names
                # s is stored as "\u0027", in python, it will be represented as "\\u0027" and not recognized as an
                # unicode, should do .encode().decode("unicode-escape") to output "\'"
                if len(splits) >= 4:
                    splits[3] = splits[3].encode().decode("unicode-escape")

                # 1. add tokens
                tokens.append(splits[0])

                # 2. add ner_tags
                if len(splits) == 1:
                    ner_tags.append('O')
                else:
                    if splits[1] == 'B':
                        ner_tags.append('B')
                    else:
                        assert splits[1] == 'I' and len(ner_tags) > 0 and ner_tags[-1] in ['B', 'I']
                        ner_tags.append('I')

                # 3. add entity_names and entity ids
                if len(splits) == 1:
                    entity_ids.append(_EMPTY_ENTITY_ID)
                    entity_names.append(_EMPTY_ENTITY_NAME)
                elif len(splits) == 4:
                    entity_ids.append(_UNK_ENTITY_ID)
                    entity_names.append(_UNK_ENTITY_NAME)
                else:
                    entity_ids.append(int(splits[5]))
                    entity_names.append(splits[3])

                    wiki_url = splits[4]
                    wiki_url_pre = "http://en.wikipedia.org/wiki/"
                    assert wiki_url.startswith(wiki_url_pre) and wiki_url[len(wiki_url_pre):] == splits[3]

        # last example
        if tokens:
            assert doc_name != ""
            if key != 'train' and key not in doc_name:
                pass
            else:
                pass
                '''
                yield guid, {
                    "id": str(guid),
                    "doc_name": doc_name,
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                    "entity_ids": entity_ids,
                    "entity_names": entity_names,
                }
                '''

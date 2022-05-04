import spacy

nlp = spacy.load('en_core_web_sm')



def local_named_entity_recognition(document):
    out_dict = {
        'entities': []
    }
    doc = nlp(document)
    for ent in doc.ents:
        #avoidhaving to add metadata, which is used in preprocessing lateron -> spacy does not provide this like the GooogleAPI
        lbl = ent.label_
        if lbl in ['MONEY', 'NUMBER', 'DATE']:
            lbl = 'PERSON'

        ent_dict = {
            'name': ent.text,
            'type': lbl
        }
        out_dict['entities'].append(ent_dict)


    return out_dict


if __name__ == "__main__":
    result = local_named_entity_recognition("Find job id and date of hire for those employees who was hired between November 5th, 2007 and July 5th, 2009.")
    if result:
        print(result)

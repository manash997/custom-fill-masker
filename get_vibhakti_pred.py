from tokenizer_and_model import model,tokenizer
import codecs
import os,subprocess
import re
def getLines(file,splitby):
    with codecs.open(file, 'r', encoding="utf-8", errors='ignore') as filer:
                    data = filer.read()
                    data = re.sub('\t+', '\t', data)
    data = data.strip('\n')
    lines = data.split(splitby)
    return lines
###Creating a post-position dictionary
postp_dict={0: ['आगे', 'अलावा', 'अंदर', 'अनुरुप', 'अनुसार', 'अन्तर्गत', 'अन्दर', 'आसपास', 'बाहर', 'बाद', 'बीच', 'भीतर', 'चलते', 'एकसाथ', 'गिर्द', 'हेतु', 'जैसा', 'जैसे', 'जैसी', 'का', 'कारण', 'करीब', 'के', 'की', 'को', 'लायक', 'लिए', 'लिये', 'मे', 'में', 'नज़दीक', 'ने', 'नें', 'नीचे', 'निकट', 'नुमा', 'ओर', 'पहले', 'पर', 'परे', 'परिणामस्वरूप', 'पर्यंत', 'पास', 'पश्चात्', 'पे', 'पीछे', 'फॉर', 'पू.', 'पूर्व', 'रूप', 'सहारे', 'सहित', 'सामने', 'समय', 'साथ', 'साथसाथ', 'सदृश', 'से', 'स्वरुप', 'स्वरूप', 'ऊपर', 'उधर', 'ऊँचाई', 'वजह', 'वाला', 'वाले', 'वाली', 'वालों', 'तहत', 'तक', 'तरफ', 'दौरान', 'दूर', 'दूरदराज़', 'दूरदूर', 'दूरी', 'द्वारा', 'योग्य'], 1: ['के', 'की', 'पीछे', 'से', 'में', 'वाले', 'तक', 'अंदर', 'अतिरिक्त', 'लिए', 'ओर', 'आगे', 'अलावा', 'अंर्तगत', 'अंतर्गत', 'अनुकूल', 'अनुसार', 'अन्तर्गत', 'अन्दर', 'आसपास', 'आधार', 'अधीन', 'बहाने', 'बाहर', 'बजाय', 'बराबर', 'बारे', 'बावजूद', 'बाद', 'बीच', 'बीचोंबीच', 'भीतर', 'चलते', 'इर्द', 'जरिए', 'जरिये', 'कारण', 'करीब', 'खिलाफ', 'किनारे', 'क्रमानुसार', 'लगभग', 'लिहाज', 'लिये', 'मध्य', 'माध्यम', 'मुकाबले', 'मुताबिक', 'नजदीक', 'नाते', 'नीचे', 'निकट', 'पहले', 'फलस्वरूप', 'परिणाम', 'परिणामस्वरूप', 'पास', 'पश्चात्', 'पश्चात', 'प्रति', 'पूर्व', 'रूप', 'सहारे', 'समान', 'समानांतर', 'सामने', 'समय', 'संबंध', 'समीप', 'समीपस्थ', 'साथ', 'साथसाथ', 'ऊपर', 'उपरान्त', 'विपरीत', 'विरुद्ध', 'विरूद्ध', 'तहत', 'तौर', 'दरम्यान', 'दौरान', 'द्वारा', 'योग्य', 'अपेक्षा', 'बजाए', 'भाँति', 'जगह', 'जैसी', 'वजह', 'तरह', 'तरफ', 'तुलना', 'दृष्टि', 'का', 'इतर', 'लेकर', 'दूर', 'पर', 'को'], 2: ['ओर', 'को', 'से', 'की', 'का', 'के', 'पर', 'में', 'गिर्द', 'स्वरूप', 'तक', 'साथ', 'मुकाबले', 'तरह', 'रूप', 'समक्ष', 'लिए'], 3: ['में']}
### function to get mask vibhakti index.
def get_mask_vib_index(indices):
  prev_ix = indices[0]
  ix_list = [0]
  for ix in indices[1:]:
    if prev_ix + 1 == ix:
      ix_list.append(1)
    else:
      ix_list.append(0)
    prev_ix = ix
  return ix_list
get_mask_vib_index([1,4,5,7])

###
import sys
class item :
    sent = ''
    probability = 0
class pred_q :
    pr = [None] * (100000)
    size = -1
    @staticmethod
    def enqueue( sent,  probability) :
        pred_q.size += 1
        pred_q.pr[pred_q.size] = item()
        pred_q.pr[pred_q.size].sent = sent
        pred_q.pr[pred_q.size].probability = probability
         
    @staticmethod
    def  peek() :
        highestProbability = -sys.maxsize
        ind = -1
        i = 0
        while (i <= pred_q.size) :
            if(highestProbability < pred_q.pr[i].probability) :
                highestProbability = pred_q.pr[i].probability
                ind = i
            i += 1
        return ind
       
    @staticmethod
    def dequeue() :
        ind = pred_q.peek()
        i = ind
        while (i < pred_q.size) :
            pred_q.pr[i] = pred_q.pr[i + 1]
            i += 1
        pred_q.size -= 1
### This code takes as input a sentence
### Encodes it using the Tokenizer
### Stores the index of masked tensor calling the create_mask_inx_tensor() function
### Calls the generate predn o/p function to get predicted o/p and scores
import torch
import itertools
import numpy as np
def create_mask_inx_tensor(encoded_input):
  
  mask_index_tensors=[]
  for val in range(len(encoded_input["input_ids"][0])):
    if encoded_input["input_ids"][0][val]==250001:
      mask_index_tensors.append(torch.tensor([val],))
  return mask_index_tensors

#----------------------------------------------------------
def generate_predn(index,mask_index,encoded_input,postp_dict):
  output = model(**encoded_input)
  #print(output)
  logits = output.logits
  input_ids=encoded_input.input_ids
  #print(logits)
  word_list=[]
  from torch.nn import functional as F
  softmax = F.softmax(logits, dim = -1)
  mask_word = softmax[0, mask_index, :]
  #-----------------------------------------------
  #print(mask_word)
  values,predictions=mask_word.topk(5)
  result = []
  single_mask = values.shape[0] == 1
  for i, (_values, _predictions) in enumerate(zip(values.tolist(), predictions.tolist())):
    row=[]
    unk_row = []
    for v, p in zip(_values, _predictions):
                # Copy is important since we're going to modify this array in place
      tokens = input_ids.numpy().copy()
      proposition = {"score": v, "token": tokenizer.decode([p])}
      if proposition["token"] in postp_dict[index]:
        row.append(proposition)
      else:
        #print(proposition["token"])
        proposition["token"] += '_unk'
        unk_row.append(proposition)
    if not row:
      result.append(unk_row)
    else:
      result.append(row)
  #print(result)
  return result
  #-----------------------------------------------
  top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
  #print(type(top_10))
  #print(top_10)
  #print("predn for token indx:{}".format(mask_index))
  for token in top_10:
    word=tokenizer.decode([token])
    word_list.append(word)
  return word_list
#-------------------------------------------------
#text=""The world will end in <mask>""
text1="अंतिम प्रकार <mask> खोजबीन <mask> <mask> हम ये सब बातें कर रहे हैं"
text2 = "मुन्नालाल <mask> <mask> दो भैंस और कुछ मुर्गियाँ भी हैं ।"
text3 = "मृदा <mask> अधिक उपजाऊ बनाने <mask> <mask> अभी खाद डाला है ।"
text4 = "क्या आप अपने क्षेत्र <mask> पाए जाने <mask> पाँच पक्षियों <mask> नाम बता सकते हैं?"
text5 = "अनंत जन्मों <mask> जिस पूर्ण पुरुषोत्तम <mask> वे ढूँढते रहें वह इस जन्म <mask> उनके ही देह <mask> प्रगट हो गये"
def fill_mask(text):
  try:
    text = text.replace('<mask><mask>','<mask> <mask>')
    text = text.replace('  ',' ')
    mtoken_ind = [m.start() for m in re.finditer('<mask>', text)]
    vib_index = get_mask_vib_index(mtoken_ind)
    encoded_input = tokenizer(text, return_tensors='pt')
    mask_index_tensors=create_mask_inx_tensor(encoded_input)
  #print(encoded_input)
  #print("text:",text)
    predn_op=[]
  #print(mask_index_tensors)
    for index in range(len(mask_index_tensors)):
      predn_op.append((generate_predn(vib_index[index],mask_index_tensors[index],encoded_input,postp_dict)))
    #print(predn_op[-1])
    #print("for token_inx:{}".format(mask_index))'
    top_3_pred = []
    #if len(predn_op) == 1:
      #top_3_pred = predn_op[0][0][0]["token"]
    all_list = [l[0] for l in predn_op]
    permutations = list(itertools.product(*all_list))
    for ps in permutations:
      tokens = [p["token"] for p in ps]
      scores = [p["score"] for p in ps]
      sent = ' '.join(tokens)
      prob = np.prod(scores)
      #print([sent,prob])
      pred_q.enqueue(sent,prob)
    
    
    for j in range(3):
      text0 = text
      pred_masks = pred_q.pr[pred_q.peek()].sent.split(' ')
      length = len(pred_masks)
      #print([pred_masks,predn_op])
      for i in range(length):
        text0 = text0.replace('<mask>',pred_masks[i],1)
      top_3_pred.append([text0,pred_q.pr[pred_q.peek()].probability])
      pred_q.dequeue()
      
    return top_3_pred
  except:
    print(f"failed for {text}")
    return []
fill_mask(text5)

###
def get_pred_col(sent):
  #sent = row['masked']
  if '<mask>' in sent:
    pred_top_3 = fill_mask(sent)
    pred_top_3_sent = [i[0] for i in pred_top_3]
    return ';'.join(pred_top_3_sent)
  else:
    return sent
preds=get_pred_col("प्रत्येक जीव <mask> शिष्यपद ग्रहण करने <mask> दृष्टि जिसने पाई है वही ज्ञानी हो सकता है")
print(preds)
#get_pred_col("ज्ञानी पुरुष <mask> भक्ति <mask> कीर्तन भक्ति करना तो सर्वश्रेष्ठ भक्ति है")
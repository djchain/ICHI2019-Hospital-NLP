# -*- coding: utf-8 -*-
"""
Created on MAR 20 22:00:24 2018
This script helps pre-preprocess data before training
@author: Ruiyu Zhang
@version: 20180328,20180403,20180404,20180405,20180415,20180420,20180424,20181017
"""
'''
Hospital Operation Dictionary Format:
Before add_voice():{(file_name,sentence_num):['sentence','label1','label2',['label3']]}
After  add_voice():{(file_name,sentence_num):[sentence_words],'label1','label2',['label3'],voice_left,voice_right]}

Word Dioctionary Format:
{'word':[num,[array from word2vec]]}

Path Format:
PLZ put dict in path/analyze/
PLZ put audio in path/L/ and path/R/ respectively
PLZ put xlsx data in path/
PLZ make sure in xlsx files, labels are in column G
'''
'''
NOTE CURRENT LABEL-FILTER METHODS:
1.remove '?' and '_SPACE_' <<< HIGHLY EFFECTIVE
2.if not found, cut shorter from back and search again <<< HIGHLY EFFECTIVE
3.also search in 'Survey Category' & 'Goal' dictionary <<< HIGHLY EFFECTIVE
4.use manual-decide function (manual_label_decide()) <<< HIGHLY EFFECTIVE
5.ignore upper or lower character <<<KINDA EFFECTIVE
PRIORITY: 1,5>2>4>3
'''
import os
import pyexcel as pe
import pyexcel.ext.xls
import types
import pickle
import warnings
import nltk
import scipy.io as scio
import warnings
import numpy as np
import random
import sys
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
import time
import scipy.io as scio
from copy import  deepcopy

#reload(sys)
#sys.setdefaultencoding('utf-8')
warnings.filterwarnings("ignore")


def str_makeup(st, length):
    # modify a string to given length
    st=str(st)
    if len(st) > length:
        st = st[0:length - 1]
    while len(st) < length:
        st += ' '
    return st

def str_make0(st, length):
    st=str(st)
    if len(st) > length:
        st = st[0:length - 1]
    while len(st) < length:
        st = '0' + st
    return st

def _is_in(list, item):
    for i,v in enumerate(list):
        if v==item:return i
    return -1

def re_index(l):
    num = 0
    dic = {}
    for i in set(l):
        dic[i] = num
        num += 1
    return dic


class data():
    def __init__(self,path='./',dic_name='FINAL_Hierarchical_Dictionary.xlsx'):
        dic_name='analyze/'+dic_name
        if path[-1]!='\\':
            path+='\\'
        path = path.replace('\\', '/')
        self.path=path
        self.dic_name=dic_name
        self.dic_h,self.dic_m,self.dic_l=self._load_dic(path,dic_name)#dictionary for labeling
        self.not_registered=0
        self.unlabeled=0
        self.labeled=0
        self.data={}
        self.cval=[('NA',0),'NA','NA',['NA'],None,None]
        self.word_dic={}#dictionary to prepare for word-embedding
        self.label_dic_h={}
        self.label_dic_m={}
        self.label_dic_l={}
        self.label_dic_lower_10={}
        self.lower_10_transfer_dic = None # for debug
        self.tester_lbl_dic = None # for debug
        self.label_mode='h'
        self.sent2vec={}#use word_dic to translate sentence into vectors
        #[sentense,label:cat,label:goal,[label:general_task],[[LeftAudio]],[[RightAudio]]]
        self.xlsx_list=self._find_file_by_type(data_path=path, type='.xlsx')
        self.trainer=[]
        self.tester=[]
        self.refresh_train_test()
        self.voice_pad_len=602
        self.sent_pad_len=30
        self.trainer_lbl_statistics={}
        self.tester_lbl_statistics={}
        self.unclear_lbl=['NA','NULL','Other']
        self.diagnose=False
        self.diag_print=False

        self._delete_epoch(path)
        print('>>>Done: Initialization.')

    def path(self,path):
        if path[-1]!='\\':
            path+='\\'
        path = path.replace('\\', '/')
        self.path=path
        print('>>>Done: Working path change.')

    def auto_process(self, merge_unclear=False, load_audio = True):
        if self.diagnose:
            sd=open(self.path + 'analyze/SentenceDiagnose.txt', 'w')
            sd.write(str_makeup('XLSX NAME', 30) + 'NUM' + '  ' + str_makeup('GIVEN LABEL', 85) + str('SENTENCE' + '\n\n'))
        path,done=self.path,0
        for xlsx_name in self.xlsx_list:
            #print('>Handling:'+ xlsx_name)
            done+=1
            sys.stdout.write('\r>{}/{} Loading: '.format(done,len(self.xlsx_list))+str(xlsx_name))
            sht = pe.get_sheet(file_name=path + xlsx_name + '.xlsx')
            whole_data = sht.to_array()[1:]
            for row_num in range(len(whole_data)):
                sent, gen_lbl = self._remove_biaodian(whole_data[row_num][4]), whole_data[row_num][6]
                cval = self._label_decide(gen_lbl)
                cval[0] = self._sentence_clip(sent)
                if not len(cval[0]):continue#skip empty sentences
                #key = xlsx_name + '-' + str_make0(str(row_num + 1), 4)
                key=(xlsx_name,row_num)#num start from 0
                if self.diagnose and self.diag_print:
                    sd.write(str_makeup(xlsx_name,30)+str_make0(row_num,3)+'  '+str_makeup(gen_lbl,85)+str(whole_data[row_num][4])+'\n')
                    self.diag_print=False
                if load_audio:
                    cval[4],cval[5]=self._read_mat(self._key_adapt(key))
                else: # skip audio loading
                    cval[4], cval[5] = np.zeros(shape = (1,1)), np.zeros(shape = (1,1))
                # data=addto_dic(key,cval,data)
                self.data[key] = cval
        sys.stdout.write('\r>>>Done: Load all scripts.\n')
        if merge_unclear:
            self._merge_unclear_labels()
        if self.diagnose:
            sd.close()
        #print('>>>Done: Load all scripts.')
        self.pre_embed()
        self._print()
        self.statistics()
        #save_dic(data)

    def _print(self):
        data=self.data
        #not_registered, unlabeled, labeled=self.not_registered,self.unlabeled,self.labeled
        i = ''
        f = open(self.path + 'analyze/DataMap.txt', 'w')
        f.write('DATA SET SIZE= ' + str(len(data)) + '\n')
        f.write('NOT REGISTERED= ' + str(self.not_registered) + '\n')
        f.write('NOT PRE-LABELED= ' + str(self.unlabeled) + '\n')
        f.write('NOW LABELED= ' + str(self.labeled) + '\n\n')
        f.write(str_makeup('SEQUENCE', 40))
        f.write(str_makeup('LBL1:SurveyCategory', 25))
        f.write(str_makeup('LBL2:Goal', 35))
        f.write(str_makeup('LBL3:GeneralTask', 35))
        f.write(str_makeup('VOICE ADDED',15))
        f.write('ACTUAL SENTENCE\n\n')
        # for key,val in dic.items():
        for key in sorted(data.keys()):
            val = data[key]
            voice_added=''
            f.write(str_makeup(str(key), 40))
            f.write(str_makeup('[' + str(val[1]) + ']', 25))
            f.write(str_makeup('[' + str(val[2]) + ']', 35))
            f.write(str_makeup(str(val[3]), 35))
            if type(val[4]) !=type('NA'):
                voice_added+='Left '
            if type(val[4]) != type('NA'):
                voice_added+='Right '
            if len(voice_added)<2:
                voice_added='NO VOICE'
            f.write(str_makeup(voice_added,15))
            f.write(str(val[0]))
            f.write('\n')
        print('>>>Done: Print data in \'DataMap.txt\'.')
        f.close()

    def update(self,key,val):#addto_dic
        if type(key) != type(('1',2)):
            print('>Error: key type is not tuple. Current object skipped.')
            return False
        if type(val) != type([1, 1]):
            print('>Error: value type not list. Current object skipped.')
            return False
        # val=tuple(val)
        self.data.update({key, val})
        print('>Updated:', key)
        return True

    def statistics(self):
        l1,l2,l3={},{},{}
        lbl1, lbl2, lbl3 = [], [], []
        count1, count2, count3 = [], [], []
        for lbl in self.data.values():
            isin1, isin2 = _is_in(lbl1, lbl[1]), _is_in(lbl2, lbl[2])
            # print(isin1,isin2,isin3)
            if isin1 >= 0:
                count1[isin1] += 1
            else:
                lbl1.append(lbl[1])
                count1.append(1)
            if isin2 >= 0:
                count2[isin2] += 1
            else:
                lbl2.append(lbl[2])
                count2.append(1)
            for l in lbl[3]:  # label3
                isin3 = _is_in(lbl3, l)
                if isin3 >= 0:
                    count3[isin3] += 1
                else:
                    lbl3.append(l)
                    count3.append(1)
        #keep into dictionaries
        for i in range(len(lbl1)):l1[lbl1[i]]=count1[i]
        for i in range(len(lbl2)):l2[lbl2[i]]=count2[i]
        for i in range(len(lbl3)):l3[lbl3[i]]=count3[i]
        f = open(self.path + 'analyze/Statistics.txt', 'w')
        f.write('LABEL 1:Survey Category\n' + str_makeup('LABEL1', 30) + 'COUNT\n')
        #for i in range(len(lbl1)):f.write('\n' + str_makeup(str(lbl1[i]), 30) + str(count1[i]))
        for k,v in sorted(l1.items(),key=lambda item:item[1],reverse=True):
            f.write('\n' + str_makeup(str(k), 30) + str(v))
        f.write('\n'*5+'_' * 50+'\nLABEL 2:Goal\n' + str_makeup('LABEL2', 30) + 'COUNT\n')
        #for i in range(len(lbl2)):f.write('\n' + str_makeup(str(lbl2[i]), 30) + str(count2[i]))
        for k,v in sorted(l2.items(),key=lambda item:item[1],reverse=True):
            f.write('\n' + str_makeup(str(k), 30) + str(v))
        f.write('\n'*5+'_' * 50+'\nLABEL 3:General Task\n' + str_makeup('LABEL3', 30) + 'COUNT\n')
        #for i in range(len(lbl3)):f.write('\n' + str_makeup(str(lbl3[i]), 30) + str(count3[i]))
        for k,v in sorted(l3.items(),key=lambda item:item[1],reverse=True):
            f.write('\n' + str_makeup(str(k), 30) + str(v))
        f.close()
        print('>>>Done: Print statisctics in \'Statistics.txt\'.')

    def save(self):
        f=open(self.path + 'analyze/Save.txt','wb')
        pickle.dump(self.data,f)
        f.close()
        print('>>>Done: Save data in Save.txt')

    def load(self):
        path=self.path
        if path!=self.path:
            try:
               f=open(path+'save.txt','rb')
            except IOError:
                print('>!Failed to load saves: not found.')
                return
        else:
            try:
                f=open(self.path + 'analyze/save.txt','rb')
            except IOError:
                print('>!Failed to load saves: not found.')
                return
        self.data=pickle.load(f)
        f.close()
        print('>>>Done: Load data from saves.')
        self.pre_embed()

    def goal_n_sec(self):
        #returns a dic of goal-label, if label is Secondary Survey, return General Task
        goal={}
        for (key,list) in self.data.items():
            label=str(list[2])
            if label.find('Secondary Survey')!=-1:
                label=list[3]
            else:
                label=[label]
            goal[key]=label
        return goal

    def data2vec(self):
        #manually renew word_dictionary
        '''
        import word2vec
        f = open(os.path.join(self.path + 'analyze/glove.6B.200d.txt'))
        embeddings_index = {}
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        '''
        #word->vec
        self.word_dic={}#clear old dic
        for row in self.data.values():
            sent=row[0]
            for word in sent:
                if word not in self.word_dic.keys():
                    tag=len(self.word_dic)+1
                    #embedding_vector = embeddings_index.get(word)
                    #if embedding_vector is None:
                    #    embedding_vector=np.zeros((1, 200))
                    self.word_dic[word]=tag
        print('>>>Done: Update word_dic. New dic size='+str(len(self.word_dic)))

        #label->vec
        self.label_dic_h={}
        self.label_dic_m={}
        self.label_dic_l={}
        for row in self.data.values():
            h,m,l=row[1],row[2],row[3]
            if h not in self.label_dic_h.keys():
                self.label_dic_h[h]=len(self.label_dic_h)
            if m not in self.label_dic_m.keys():
                self.label_dic_m[m]=len(self.label_dic_m)
            for l_ in l:
                if l_ not in self.label_dic_l.keys():
                    self.label_dic_l[l_]=len(self.label_dic_l)
        print('>>>Done: Update label_dic.')
        self._print_data2vec()

    def set_mode(self,mode):
        if mode=='h':
            self.label_mode='h'
            print('>>>Done: Set mode to \'h\'.')
        elif mode=='m':
            self.label_model='m'
            print('>>>Done: Set mode to \'m\'.')
        elif mode=='l':
            self.label_model='l'
            print('>>>Done: Set mode to \'l\'.')
        else:
            print('>>>!Faild to set mode. Current mode= \''+self.label_mode+'\'.')

    def pre_embed(self):
        #updates self.sent2vec
        bed={}#creat new
        self.data2vec()
        self.sent2vec={}#clear
        for key, val in self.data.items():
            sent=list(val[0])
            for i,word in enumerate(sent):
                if word not in self.word_dic.keys():
                    self.word_dic[word]=len(self.word_dic)+1
                    print('>Warning: One word not found in dict. Solved by updating word_dic.')
                sent[i] = self.word_dic[word]
            self.sent2vec[key]=sent
        print('>>>Done: Updated word embedded sentences.')

    def rename_label_dic(self,new_name):
        #manually set dictionary name
        new_name=str(new_name)
        self.dic_name='analyze/'+new_name
        print('>>>Done: Dictionary changed to [',new_name,'].')

    def refresh_train_test(self,pivot_percent=0.75):
        pivot=int(len(self.xlsx_list)*pivot_percent)
        random.shuffle(self.xlsx_list)
        self.trainer=self.xlsx_list[:pivot]
        self.tester=self.xlsx_list[pivot:]
        print('>>>Done: Refresh train/test sets.\n>Training set='+str(len(self.trainer))+'  Testing set='+str(len(self.tester)))

    def get_trainer(self,average=False,average_size=60,sort=False):
        #return sentence with label, training set and testing set
        #label_vec or [label_vec],[sentence_vec],arr[L_voice],arr[R_voice]
        lbl,sent,L,R=[],[],[],[]
        if self.label_mode=='h':
            lbl_dic=self.label_dic_h
            lbl_tag=1
            print('>Label hierarchy: High')
        elif self.label_mode=='m':
            lbl_dic=self.label_dic_m
            lbl_tag = 2
            print('>Label hierarchy: Midium')
        elif self.label_mode=='l':
            lbl_dic=self.label_dic_l
            lbl_tag = 3
            print('>Label hierarchy: Low')
        elif self.label_mode=='lower_10':
            print('>!Warning: lower 10 label mode is adopted')
            lbl_dic=self.label_dic_l
            lbl_tag = 3
            tar_list = ['Back', 'GCS Calculation', 'Oxygen', 'Head', 'C-Spine', 'Pulse Check', 'Blood Pressure',
                        'Extremity', 'Mouth', 'Abdomen','NULL']
            if 'NULL' not in lbl_dic:
                lbl_dic['NULL']=len(lbl_dic)
            for l, i in lbl_dic.items():
                if l not in tar_list:
                    lbl_dic[l]=lbl_dic['NULL']#不在上面要求的十个之列，则置空
        else:
            print('>!Error getting trainers: Unidentified label_mode. Returning NULL data.')
            return lbl,sent,L,R
        if sort:
            for key,val in sorted(self.data.items(),key=lambda item:item[0]):
                if key[0] in self.trainer:
                    if lbl_tag < 3:
                        lbl.append(lbl_dic[val[lbl_tag]])
                    else:
                        lbl.append(lbl_dic[val[lbl_tag][0]])
                    sent.append(self.sent2vec[key])
                    L.append(val[4].transpose())
                    R.append(val[5].transpose())
        else:
            keys=list(self.data.keys())
            random.shuffle(keys)
            for key in keys:
                val=self.data[key]
                if key[0] in self.trainer and val[4]!='NA' and val[5]!='NA':
                    if lbl_tag < 3:
                        lbl.append(lbl_dic[val[lbl_tag]])
                    else:
                        lbl.append(lbl_dic[val[lbl_tag][0]])
                    sent.append(self.sent2vec[key])
                    L.append(val[4].transpose())
                    R.append(val[5].transpose())

        if average:
            #lbl,sent,L,R=self._batch_average(lbl,sent,L,R)
            lbl,sent,L,R=self._ramdon_select_from_trainertester(lbl,sent,L,R,each_count=average_size)
        self._train_test_statistics(lbl, lbl_dic,'train')
        L = sequence.pad_sequences(np.array(L),dtype='float32', padding='pre', maxlen=self.voice_pad_len, value=0.)
        R = sequence.pad_sequences(np.array(R),dtype='float32', padding='pre', maxlen=self.voice_pad_len, value=0.)
        sent =sequence.pad_sequences(sent,dtype='float32', padding='pre', maxlen=self.sent_pad_len)
        if self.label_mode=='lower_10':
            if not self.label_dic_lower_10:
                self.lable_dic_lower_10 = re_index(lbl)
            lbl = [self.lable_dic_lower_10[x] for x in lbl]
            lbl = to_categorical(lbl, num_classes=11)
            # print('train label dic: ', self.lable_dic_lower_10)
        else:
            lbl = to_categorical(lbl, num_classes=len(lbl_dic))
        return lbl,sent,L,R

    def get_tester(self,average=False,average_size=60,debug=False):
        #return sentence with label, training set and testing set
        #label_vec or [label_vec],[sentence_vec],arr[L_voice],arr[R_voice]
        lbl,sent,L,R=[],[],[],[]
        if self.label_mode=='h':
            lbl_dic=self.label_dic_h
            lbl_tag=1
        elif self.label_mode=='m':
            lbl_dic=self.label_dic_m
            lbl_tag = 2
        elif self.label_mode=='l':
            lbl_dic=self.label_dic_l
            lbl_tag = 3
        elif self.label_mode == 'lower_10':
            print('>!Warning: lower 10 label mode is adopted')
            lbl_dic = self.label_dic_l
            lbl_tag = 3
            tar_list = ['Back', 'GCS Calculation', 'Oxygen', 'Head', 'C-Spine', 'Pulse Check', 'Blood Pressure',
                        'Extremity', 'Mouth', 'Abdomen', 'NULL']
            if 'NULL' not in lbl_dic:
                lbl_dic['NULL']=len(lbl_dic)
            for l, i in lbl_dic.items():
                if l not in tar_list:
                    lbl_dic[l] = lbl_dic['NULL']  # 不在上面要求的十个之列，则置空
                #else:print('TEST: 10 LABEL FILTER WORKS')
        else:
            print('>!Error getting testers: Unidentified label_mode. Returning NULL data.')
            return lbl,sent,L,R
        for key,val in sorted(self.data.items(),key=lambda item:item[0]):
            if key[0] in self.tester and val[4]!='NA' and val[5]!='NA':
                if lbl_tag<3:
                    lbl.append(lbl_dic[val[lbl_tag]])
                else:
                    lbl.append(lbl_dic[val[lbl_tag][0]])
                sent.append(self.sent2vec[key])
                L.append(val[4].transpose())
                R.append(val[5].transpose())
        if average:
            lbl,sent,L,R=self._ramdon_select_from_trainertester(lbl,sent,L,R,each_count=average_size)
        self._train_test_statistics(lbl,lbl_dic, 'test')
        L = sequence.pad_sequences(np.array(L),dtype='float32', maxlen=self.voice_pad_len)
        R = sequence.pad_sequences(np.array(R),dtype='float32', maxlen=self.voice_pad_len)
        sent = sequence.pad_sequences(sent,dtype='float32', maxlen=self.sent_pad_len)
        if self.label_mode=='lower_10':
            if not self.label_dic_lower_10:
                self.lable_dic_lower_10 = re_index(lbl)
                if debug:
                    self.lower_10_transfer_dic = re_index(lbl)
                    self.tester_lbl_dic = lbl_dic
            lbl = [self.lable_dic_lower_10[x] for x in lbl]
            lbl = to_categorical(lbl, num_classes=11)
            # print('test label dic: ', self.lable_dic_lower_10)
        else:
            lbl = to_categorical(lbl, num_classes=len(lbl_dic))
        return lbl,sent,L,R

    def get_data_for_analyze(self, label = None, traintest = None):
        # get data from both trainer and tester, with data including both feature and raw data
        lbl, sent, L, R, raw_lbl, raw_sent = [], [], [], [], [], []
        if self.label_mode=='h': lbl_dic, lbl_tag = self.label_dic_h, 1
        elif self.label_mode=='m': lbl_dic, lbl_tag = self.label_dic_m, 2
        elif self.label_mode=='l': lbl_dic, lbl_tag = self.label_dic_l, 3
        elif self.label_mode == 'lower_10':
            print('>!Warning: lower 10 label mode is adopted')
            lbl_dic, lbl_tag = self.label_dic_l, 3
            tar_list = ['Back', 'GCS Calculation', 'Oxygen', 'Head', 'C-Spine', 'Pulse Check', 'Blood Pressure',
                        'Extremity', 'Mouth', 'Abdomen', 'NULL']
            if 'NULL' not in lbl_dic: lbl_dic['NULL'] = len(lbl_dic)
            for l, i in lbl_dic.items():
                if l not in tar_list:
                    lbl_dic[l] = lbl_dic['NULL']  # 不在上面要求的十个之列，则置空
                #else:print('TEST: 10 LABEL FILTER WORKS')
        else:
            print('>!Error getting testers: Unidentified label_mode. Returning NULL data.')
            return lbl, sent, L, R, raw_lbl, raw_sent
        # Preprocess
        for key, val in sorted(self.data.items(),key=lambda item:item[0]):
            l = val[lbl_tag] if lbl_tag < 3 else val[lbl_tag][0]
            if label and label != l: continue # if label filter "var label" is active
            lbl.append(lbl_dic[l])
            raw_lbl.append(l)
            sent.append(self.sent2vec[key])
            raw_sent.append(val[0])
            L.append(val[4].transpose())
            R.append(val[5].transpose())
        L = sequence.pad_sequences(np.array(L),dtype='float32', maxlen=self.voice_pad_len)
        R = sequence.pad_sequences(np.array(R),dtype='float32', maxlen=self.voice_pad_len)
        sent = sequence.pad_sequences(sent,dtype='float32', maxlen=self.sent_pad_len)
        if self.label_mode=='lower_10':
            if not self.label_dic_lower_10:
                self.lable_dic_lower_10 = re_index(lbl)
            lbl = [self.lable_dic_lower_10[x] for x in lbl]
            lbl = to_categorical(lbl, num_classes=11)
            # print('test label dic: ', self.lable_dic_lower_10)
        else:
            lbl = to_categorical(lbl, num_classes=len(lbl_dic))
        return lbl, sent, L, R, raw_lbl, raw_sent


    def get_raw_trainer(self,average=False,average_size=60,sort=False):
        #return sentence with label, training set and testing set, not piled for training
        #label_vec or [label_vec],[sentence_vec],arr[L_voice],arr[R_voice]
        lbl,sent,L,R=[],[],[],[]
        if self.label_mode=='h':
            lbl_dic=self.label_dic_h
            lbl_tag=1
            print('>Label hierarchy: High')
        elif self.label_mode=='m':
            lbl_dic=self.label_dic_m
            lbl_tag = 2
            print('>Label hierarchy: Midium')
        elif self.label_mode=='l':
            lbl_dic=self.label_dic_l
            lbl_tag = 3
            print('>Label hierarchy: Low')
        elif self.label_mode=='lower_10':
            print('>!Warning: lower 10 label mode is adopted')
            lbl_dic=self.label_dic_l
            lbl_tag = 3
            tar_list = ['Back', 'GCS Calculation', 'Oxygen', 'Head', 'C-Spine', 'Pulse Check', 'Blood Pressure',
                        'Extremity', 'Mouth', 'Abdomen','NULL']
            if 'NULL' not in lbl_dic:
                lbl_dic['NULL']=len(lbl_dic)
            for l,i in  lbl_dic.items():
                if l not in tar_list:
                    lbl_dic[l]=lbl_dic['NULL']#不在上面要求的十个之列，则置空
        else:
            print('>!Error getting trainers: Unidentified label_mode. Returning NULL data.')
            return lbl,sent,L,R
        if sort:
            for key,val in sorted(self.data.items(),key=lambda item:item[0]):
                if key[0] in self.trainer:
                    if lbl_tag < 3:
                        lbl.append(lbl_dic[val[lbl_tag]])
                    else:
                        lbl.append(lbl_dic[val[lbl_tag][0]])
                    sent.append(self.sent2vec[key])
                    L.append(val[4].transpose())
                    R.append(val[5].transpose())
        else:
            keys=list(self.data.keys())
            random.shuffle(keys)
            for key in keys:
                val=self.data[key]
                if key[0] in self.trainer and val[4]!='NA' and val[5]!='NA':
                    if lbl_tag < 3:
                        lbl.append(lbl_dic[val[lbl_tag]])
                    else:
                        lbl.append(lbl_dic[val[lbl_tag][0]])
                    sent.append(self.sent2vec[key])
                    L.append(val[4].transpose())
                    R.append(val[5].transpose())

        if average:
            #lbl,sent,L,R=self._batch_average(lbl,sent,L,R)
            lbl,sent,L,R=self._ramdon_select_from_trainertester(lbl,sent,L,R,each_count=average_size)
        self._train_test_statistics(lbl, lbl_dic,'train')
        '''
        L = sequence.pad_sequences(np.array(L),dtype='float32', padding='pre', maxlen=self.voice_pad_len, value=0.)
        R = sequence.pad_sequences(np.array(R),dtype='float32', padding='pre', maxlen=self.voice_pad_len, value=0.)
        sent =sequence.pad_sequences(sent,dtype='float32', padding='pre', maxlen=self.sent_pad_len)
        if self.label_mode=='lower_10':
            if not self.label_dic_lower_10:
                self.lable_dic_lower_10 = re_index(lbl)
            lbl = [self.lable_dic_lower_10[x] for x in lbl]
            lbl = to_categorical(lbl, num_classes=11)
            # print('train label dic: ', self.lable_dic_lower_10)
        else:
            lbl = to_categorical(lbl, num_classes=len(lbl_dic))
        '''
        return lbl,sent,L,R

    def get_raw_tester(self,average=False,average_size=60):
        #return sentence with label, training set and testing set, not piled for training
        #label_vec or [label_vec],[sentence_vec],arr[L_voice],arr[R_voice]
        lbl,sent,L,R=[],[],[],[]
        if self.label_mode=='h':
            lbl_dic=self.label_dic_h
            lbl_tag=1
        elif self.label_mode=='m':
            lbl_dic=self.label_dic_m
            lbl_tag = 2
        elif self.label_mode=='l':
            lbl_dic=self.label_dic_l
            lbl_tag = 3
        elif self.label_mode == 'lower_10':
            print('>!Warning: lower 10 label mode is adopted')
            lbl_dic = self.label_dic_l
            lbl_tag = 3
            tar_list = ['Back', 'GCS Calculation', 'Oxygen', 'Head', 'C-Spine', 'Pulse Check', 'Blood Pressure',
                        'Extremity', 'Mouth', 'Abdomen', 'NULL']
            if 'NULL' not in lbl_dic:
                lbl_dic['NULL']=len(lbl_dic)
            for l, i in lbl_dic.items():
                if l not in tar_list:
                    lbl_dic[l] = lbl_dic['NULL']  # 不在上面要求的十个之列，则置空
                #else:print('TEST: 10 LABEL FILTER WORKS')
        else:
            print('>!Error getting testers: Unidentified label_mode. Returning NULL data.')
            return lbl,sent,L,R
        for key,val in sorted(self.data.items(),key=lambda item:item[0]):
            if key[0] in self.tester and val[4]!='NA' and val[5]!='NA':
                if lbl_tag<3:
                    lbl.append(lbl_dic[val[lbl_tag]])
                else:
                    lbl.append(lbl_dic[val[lbl_tag][0]])
                sent.append(self.sent2vec[key])
                L.append(val[4].transpose())
                R.append(val[5].transpose())
        if average:
            lbl,sent,L,R=self._ramdon_select_from_trainertester(lbl,sent,L,R,each_count=average_size)
        self._train_test_statistics(lbl,lbl_dic, 'test')
        '''
        L = sequence.pad_sequences(np.array(L),dtype='float32', maxlen=self.voice_pad_len)
        R = sequence.pad_sequences(np.array(R),dtype='float32', maxlen=self.voice_pad_len)
        sent = sequence.pad_sequences(sent,dtype='float32', maxlen=self.sent_pad_len)
        if self.label_mode=='lower_10':
            if not self.label_dic_lower_10:
                self.lable_dic_lower_10 = re_index(lbl)
            lbl = [self.lable_dic_lower_10[x] for x in lbl]
            lbl = to_categorical(lbl, num_classes=11)
            # print('test label dic: ', self.lable_dic_lower_10)
        else:
            lbl = to_categorical(lbl, num_classes=len(lbl_dic))
        '''
        return lbl,sent,L,R

    def get_traintest_script(self,filename):
        #modified version of get_trainer, returns data from a random script
        #default is sorted
        lbl, sent, L, R = [], [], [], []
        if filename not in self.xlsx_list:
            print('>!>ERROR: Given file not among script list.')
            return lbl,sent,L,R
        if self.label_mode=='h':
            lbl_dic=self.label_dic_h
            lbl_tag=1
            print('>Label hierarchy: High')
        elif self.label_mode=='m':
            lbl_dic=self.label_dic_m
            lbl_tag = 2
            print('>Label hierarchy: Midium')
        elif self.label_mode=='l':
            lbl_dic=self.label_dic_l
            lbl_tag = 3
            print('>Label hierarchy: Low')
        else:
            print('>!Error getting trainers: Unidentified label_mode. Returning NULL data.')
            return lbl,sent,L,R
        for key, val in sorted(self.data.items(), key=lambda item: item[0]):
            if key[0] == filename:
                lbl.append(lbl_dic[val[lbl_tag]])
                sent.append(self.sent2vec[key])
                val[4] = val[4].transpose()
                val[5] = val[5].transpose()
                L.append(val[4])
                R.append(val[5])
        #self._train_test_statistics(lbl, lbl_dic, 'train')
        L = sequence.pad_sequences(np.array(L), dtype='float32', padding='pre', maxlen=self.voice_pad_len, value=0.)
        R = sequence.pad_sequences(np.array(R), dtype='float32', padding='pre', maxlen=self.voice_pad_len, value=0.)
        sent = sequence.pad_sequences(sent, dtype='float32', padding='pre', maxlen=self.sent_pad_len)
        lbl = to_categorical(lbl, num_classes=len(lbl_dic))
        return lbl, sent, L, R

    def set_padding(self,sentence=30,voice=602):
        self.sent_pad_len,self.voice_pad_len=sentence,voice
        print('>>>Done: Set padding parameters.')

    def get_embed_matrix(self):
        dic = self.word_dic
        path = self.path + 'analyze/'
        embeddings_index = {}
        f = open(os.path.join(path, 'glove.6B.200d.txt'),encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('>>>Found %s word vectors.' % len(embeddings_index))

        embedding_matrix = np.zeros((len(dic) + 1, 200))
        for word, i in dic.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        # em_text = Embedding(len(dic) + 1, 200, weights=[embedding_matrix], trainable=True)(text_input)
        return embedding_matrix

    def write_epoch_acc(self,epoch,acc,name='Text',hide=False):
        f=open(self.path+'analyze/EpochAcc_'+name+'.txt','a')
        perc=acc*100
        f.write(str_makeup('\nEpoch '+str(epoch),15)+'[')
        writer=0
        while writer<perc:
            f.write('=')
            writer+=1
        f.write(' '*(100-writer)+'] {:.5f}%'.format(perc))
        f.close()
        if not hide:
            print('>'+name+' Epoch '+str(epoch)+' acc recorded.')

    def crop(self,lbl,sent,L,R,size,start=-1,end=-1):
        #crops (randomly select) trainer/tester dataset into given size
        l2,s2,L2,R2=np.array([]),np.array([]),np.array([]),np.array([])
        #l3,s3,L3,R3=[],[],[],[]
        if end>start and start>0 and end-start>=size:
            for i in range(start,end+1):
                np.append(l2,lbl[i])
                np.append(s2,sent[i])
                np.append(L2,L[i])
                np.append(R2,R[i])
        else:
            l2,s2,L2,R2=lbl,sent,L,R
        choices=[]
        random.seed(time.time())
        while len(choices)<size:
            new_choice=random.randint(0,len(l2)-1)
            if new_choice not in choices:
                choices.append(new_choice)
        l3=np.array([l2[i]for i in choices])
        s3=np.array([s2[i]for i in choices])
        L3=np.array([L2[i]for i in choices])
        R3=np.array([R2[i]for i in choices])
        '''
        for choice in choices:
            print('>DEBUG> l2[choice]=',l2[choice])
            np.append(l3,l2[choice])
            np.append(s3,s2[choice])
            np.append(L3,L2[choice])
            np.append(R3,R2[choice])
        '''
        return l3,s3,L3,R3



    def _merge_unclear_labels(self):
        #merge unclear labels such as 'Other', 'NULL', 'NA' into 'UNCLEAR'
        for key,row in self.data.items():
            self.data[key][1] = self._unclear_label_fltr(row[1])
            self.data[key][2] = self._unclear_label_fltr(row[2])
            self.data[key][3] = [self._unclear_label_fltr(sub_lbl) for sub_lbl in row[3] ]
        print('>>>Done: Transfer all unclear labels into \'UNCLEAR\'.')

    def _unclear_label_fltr(self,lbl):
        if lbl in self.unclear_lbl:
            lbl='UNCLEAR'
        return lbl

    def _load_dic(self,path,dic_name='analyze/FINAL_Hierarchical_Dictionary.xlsx'):
        # load FINAL_Hierarchical_Dictionary.xlsx
        h, m, l = [], [], []  # high, mid, low
        sheet = pe.get_sheet(file_name=path + dic_name)
        whole_data = sheet.to_array()[1:]
        for row in whole_data:
            row[3] = row[3].replace(' (', '(')
            row[4] = row[4].replace(' (', '(')
            row[5] = row[5].replace(' (', '(')
            h.append(self._remove_kuohao(row[3]))  # column num.3
            m.append(self._remove_kuohao(row[4]))
            l.append(self._remove_kuohao(row[5]))
        f = open(self.path + 'analyze/Dictionary.txt', 'w')
        for i in range(len(h)):
            f.write(str_makeup(h[i], 30) + str_makeup(m[i], 30) + str(l[i]) + '\n')
        f.close()
        print('>>>Done: Import and save Hierarchical_Dictionary.')
        return h, m, l

    def _remove_kuohao(self,stri):
        #remove inner parentheses
        stri = str(stri)
        start, end = -1, -1
        for i in range(len(stri)):
            if stri[i] == '(':
                start = i
            if stri[i] == ')':
                end = i
        if start != -1 and end != -1:
            stri = stri[:start] + stri[end + 1:]
        return stri

    def _delete_epoch(self,path):
        file_list=os.listdir(path+'analyze/')
        for file in file_list:
            if file.find('Epoch')!=-1:
                os.remove(path+'analyze/'+file)
                print('>Done: remove logfile:',file)

    def _remove_biaodian(self,current_sentence):
        # removes punctuation
        blacklist=[',','.','!',';','?','[',']','*','…','’',':','(',')','`','\'\'']
        current_sentence = str(current_sentence)
        current_sentence = current_sentence.lower()
        for c in blacklist:
            current_sentence=current_sentence.replace(c,'')
        #current_sentence = current_sentence.replace('-', ' ')
        current_sentence=current_sentence.replace('/',' over ')
        current_sentence=current_sentence.replace('%',' percent')
        return current_sentence

    def _label_decide(self,gen_lbl):
        # decide what labels should be based on given general_task label
        # return a val[,,,]
        hi, mid, lo = self.dic_h,self.dic_m,self.dic_l  # hier dictionaries
        #not_registered, unlabeled, labeled=self.not_registered,self.unlabeled,self.labeled
        '''下面是一个临时功能，用来合并四种extrimity的，方便之后train模型'''
        if gen_lbl.find('xtremity')>0:
            return ['NA', 'Secondary Survey', 'Secondary Survey Assessment', ['Extremity'],'NA','NA']
        '''结束：临时功能'''

        if self.diagnose:
            if gen_lbl.find('?')>0 or gen_lbl.find('nclear')>0 or gen_lbl.find('robably')>0:
                self.diag_print=True
        xtra_lbl = ''
        val = ['NA', 'NA', 'NA', ['NA'],'NA','NA']
        gen_lbl = gen_lbl.replace('unclear ', '')
        gen_lbl = gen_lbl.replace('Probably ', '')
        gen_lbl = gen_lbl.replace('?? ', '')
        gen_lbl = gen_lbl.replace('??', '')
        gen_lbl = gen_lbl.replace('?', '')
        _or = gen_lbl.find(' or ')  # find if concurrent
        if _or > 0:  # concurrent
            lbls=gen_lbl.split(' or ')
            gen_lbl,xtra_lbl=lbls[0],lbls[1:]
            #gen_lbl, xtra_lbl = gen_lbl[:_or], gen_lbl[_or + 4:]
        if len(gen_lbl) < 3:
            # print('>One object marks null label.')
            self.unlabeled += 1
            return ['NA', 'NULL', 'NULL', ['NULL'],'NA','NA']
        # now real work here
        index = self._dic_search(gen_lbl, lo)
        if index == -1:  # not found in lo[]
            # print('>One object not registered.')
            val_manual = self._manual_label_decide(gen_lbl)
            if val_manual[1] != -1:  # manual labeling succeeds
                self.labeled += 1
                return val_manual
            # Not yet found in lo[]
            index = self._dic_search(gen_lbl, mid)
            if index != -1:  # found in mid[]#abnormal
                self.labeled += 1
                return ['NA', hi[index], mid[index], ['NA'],'NA','NA']
            index = self._dic_search(gen_lbl, hi)
            if index != -1:  # found in hi[]#abnormal
                self.labeled += 1
                return ['NA', hi[index], 'NA', ['NA'],'NA','NA']
            # Not found finally
            self.not_registered += 1
            return val
        # found in lo[]
        val[1], val[2], val[3] = hi[index], mid[index], [lo[index]]
        if len(xtra_lbl):
            for xlbl in xtra_lbl:
                xlbl=str(xlbl)
                xlbl=xlbl.replace('c-spine','C-Spine')
                xlbl=xlbl.replace('C-spine','C-Spine')
                val[3].append(xlbl)
        self.labeled += 1
        #print (val[1])
        return val  # NOTICE: val[0] val[4] val[5] are still null

    def _manual_label_decide(self,lbl):
        # manually deal with hard-to-classify labels
        lbl = lbl.replace(' ', '')



        if lbl.find('xy') and lbl.find('reathing')!= -1:  # Oxygen
            return ['', 'Breathing', 'Breathing Control', ['Oxygen'],'NA','NA']
        if lbl.find('neumothor') != -1:  #Open Pneumo IDed
            return ['', 'Breathing', 'Breathing Control', ['Pneumothorax IDed'],'NA','NA']
        if lbl.find('horacostomy') != -1 and lbl.find('eedle') != -1:
            return ['', 'Breathing', 'Breathing Control', ['Needle Thoracostomy'],'NA','NA']
        if lbl.find('pen') != -1 and lbl.find('neumo') != -1:  # Oxygen
            return ['', 'Breathing', 'Breathing Control', ['Open Pneumo IDed'],'NA','NA']
        if lbl.find('horacostomy') != -1 and lbl.find('ube') != -1:
            return ['', 'Breathing', 'Breathing Control', ['Tube Thoracostomy'],'NA','NA']
        if lbl.find('reathing') != -1 and lbl.find('ontrol') != -1:  # Breathing Control
            return ['', 'Breathing', 'Breathing Control', ['NA'],'NA','NA']
        if lbl.find('hest') != -1 and lbl.find('isual') != -1:
            return ['', 'Breathing', 'Breathing Assessment', ['Chest Visual'],'NA','NA']
        if lbl.find('rachea') != -1 and lbl.find('isual') != -1:
            return ['', 'Breathing', 'Breathing Assessment', ['Trachea Visual'],'NA','NA']
        if lbl.find('est') != -1 and lbl.find('uscultation') != -1:
            return ['', 'Breathing', 'Breathing Assessment', ['Chest Auscultation'],'NA','NA']
        if lbl.find('ulse') != -1 and lbl.find('lacement') != -1:
            return ['', 'Breathing', 'Breathing Assessment', ['Pulse Ox Placement'],'NA','NA']
        if lbl.find('reathing') != -1 and lbl.find('ssess') != -1:  # Breathing Assessment
            return ['', 'Breathing', 'Breathing Assessment', ['NA'],'NA','NA']
        if lbl.find('irway'): # Airway Assessment
            return ['', 'Airway', 'Airway Assessment', ['NA'],'NA','NA']
        if lbl.find('xposure'):  # Exposure Assessment
            return ['', 'Exposure/Environment', 'Exposure Assessment', ['NA'], 'NA', 'NA']
        if lbl == 'LogRoll-BK' or lbl == 'LogRollBK':
            return ['', 'Secondary Survey', 'Secondary Survey Assessment', ['Back'],'NA','NA']
        if lbl == 'EarAssessment-otoscopy' or lbl == 'EarAssessment':
            return ['', 'Secondary Survey', 'Secondary Survey Assessment', ['Ears'],'NA','NA']
        if lbl.find('Arriv') != -1 or lbl.find('arriv') != -1:
            if self.diagnose:
                self.diag_print=True
            return ['', 'Other', 'Other', ['Other'],'NA','NA']
        if lbl.find('Depart') != -1 or lbl.find('depart') != -1:
            if self.diagnose:
                self.diag_print=True
            return ['', 'Other', 'Other', ['Other'],'NA','NA']
        if lbl.find('ransition') != -1 or lbl.find('ummary') != -1 or lbl.find('edication') != -1:
            # Transition, Summary, Medications
            if self.diagnose:
                self.diag_print=True
            return ['', 'Other', 'Other', ['Other'],'NA','NA']
        if lbl.find('upil') != -1:  # Pupils
            return ['', 'Disability', 'Disability Assessment', ['Pupils'],'NA','NA']
        if lbl.find('c-spine') != -1 or lbl.find('C-Spine') != -1 or lbl.find('C-spine') != -1:  # C-Spine
            return ['', 'Disability', 'Disability Control', ['C-Spine'],'NA','NA']
        if lbl.find('bolus') != -1 or lbl.find('Bolus') != -1:
            return ['', 'Circulation', 'Circulation Control', ['Bolus'],'NA','NA']
        if lbl.find('pain') != -1 and lbl.find('ssess') != -1:  # Pain Assessment
            return ['', 'Other', 'Other', ['Other'],'NA','NA']
        if lbl.find('CO2') != -1:  # CO2 monitor
            return ['', 'Airway', 'Airway Control', ['Confirm ET/Trach Tube'],'NA','NA']
        if lbl.find('enital') != -1:  # Genital Assessment
            return ['', 'Secondary Survey', 'Secondary Survey Assessment', ['Genitalia'],'NA','NA']
        if lbl.find('djunct') != -1:  # Adjuncts
            return ['', 'Secondary Survey', 'Secondary Survey Adjuncts', ['NA'],'NA','NA']
        if lbl.find('IV') != -1 and lbl.find('irculation') != -1:  # Circulation Control - IV P...
            return ['', 'Circulation', 'Circulation Control', ['IV Placement'],'NA','NA']
        if lbl.find('ulse') != -1 and lbl.find('ximetry') != -1:  # Pulse Oximetry
            return ['', 'Monitor Vital Signs', 'Oxygen Saturation', ['Pulse Oximetry'],'NA','NA']

        if lbl.find('hest') != -1 and lbl.find('ssess') != -1:  # Chest Assessment
            return ['', 'Secondary Survey', 'Secondary Survey Assessment', ['Chest'],'NA','NA']

        if lbl.find('lood') != -1 and lbl.find('ressure') != -1:  # Blood Pressure
            return ['', 'Circulation', 'Circulation Assessment', ['Blood Pressure'],'NA','NA']
        if lbl.find('Ear') != -1:  # Ear Assessment
            return ['', 'Secondary Survey', 'Secondary Survey Assessment', ['Ears'],'NA','NA']
        if lbl.find('GCS') != -1:
            return ['', 'Disability', 'Disability Assessment', ['GCS Calculation'],'NA','NA']
        if lbl.find('CPR') != -1:
            return ['', 'Circulation', 'Circulation Control', ['CPR'],'NA','NA']
        if lbl.find('EMS') != -1:
            return ['', 'Disability', 'Disability Control', ['C-Spine'],'NA','NA']

        if lbl.find('espirat') != -1:  # Respiratory
            return ['', 'Monitor Vital Signs', 'Respiratory Rate', ['Monitor Respiratory Rate'],'NA','NA']
        if lbl.find('xtremity') != -1:  # Extremity Assess with unclear label
            return ['', 'Secondary Survey', 'Secondary Survey Assessment', ['NA'],'NA','NA']
        if lbl.find('nclear') != -1 or lbl.find('onsult') != -1:  # unclear
            if self.diagnose:
                self.diag_print=True
            return ['', 'NULL', 'NULL', ['NULL'],'NA','NA']
        if self.diagnose:
            self.diag_print = True
        return ['', -1, 'NA', ['NA'],'NA','NA']

    def _dic_search(self,lbl, dic):
        # search given label in dic[]lo[], return index. if n/a return -1
        # global lo
        lbl += ' '
        while (len(lbl) > 3):
            lbl = lbl[:-1]
            for i in range(len(dic)):
                if lbl.lower() == dic[i].lower():
                    return i
        return -1

    def _label_fltr(self,label):
        # label filter
        label=str(label)
        label =label.replace(' ', '')
        label =label.replace(',', '')
        label =label.replace('unsure', '')
        return label

    def _find_file_by_type(self,data_path, type='.xlsx'):
        # find specific typed file under specific path
        file_list, name_list = os.listdir(data_path), []
        for i in file_list:
            # os.path.splitext():分离文件名与扩展名
            if os.path.splitext(i)[1] == type:
                if i[0] == '~' and i[1] == '$':  # avoid temp opened file
                    continue
                name_list.append(i[0:0 - len(type)])  # remove '.xxxx'
        return name_list

    def _sentence_clip(self,sent):
        # longStr to list of shortStr
        return nltk.word_tokenize(sent)

    def _read_mat(self,name):
        #Note: loadmat() raises TypeError(NoneType not iterable) instead of FileNotFound
        path=self.path
        try:
            if os.path.exists(path+'L/'+name+'.mat'):
                left= scio.loadmat(path+'L/'+name+'.mat')
                left=left['z1']#to array
                #print('read  mat:', path + 'L/' + name + '.mat')
            else:
                left = np.zeros(shape = (1,1))
                #print('>!',name+'.mat','not found')

            if os.path.exists(path+'R/'+name+'.mat'):
                right = scio.loadmat(path+'R/'+name+'.mat')
                right = right['z1']
            else:
                right = np.zeros(shape = (1,1))
                #print('>!',name+'.mat','not found')
        except IOError:
            print('>Voice file not loaded:',name+'.mat')
        return left,right

    def _key_adapt(self,newkey):
        #convert key type: ('filename',num) to old type: filename-000num
        old = newkey[0] + '-' + str_make0(str(newkey[1] + 1), 4)
        return old

    def _print_data2vec(self):
        f=open(self.path+'analyze/WordDic.txt','w')
        f.write('Dictionary size= '+str(len(self.word_dic))+' words\n\n')
        for word,coded in sorted(self.word_dic.items(),key=lambda item:item[1]):
            f.write(str_makeup(word,20)+str(coded)+'\n')#+str_makeup(coded[0],8)+str(coded[1])+'\n')
        f.close()
        f=open(self.path+'analyze/LabelDic.txt','w')
        f.write('Dictionary size:'+'\nHigh='+str(len(self.label_dic_h))+'\nMid='+str(len(self.label_dic_m))+'\nLow='+str(len(self.label_dic_l))+'\n\n')
        f.write('LABEL 1\n')
        for lbl,coded in sorted(self.label_dic_h.items(),key=lambda item:item[1]):
            f.write(str_makeup(lbl, 40) + str(coded) + '\n')
        f.write('\n\nLABEL 2\n')
        for lbl, coded in sorted(self.label_dic_m.items(), key=lambda item: item[1]):
            f.write(str_makeup(lbl, 40) + str(coded) + '\n')
        f.write('\n\nLABEL 3\n')
        for lbl, coded in sorted(self.label_dic_l.items(), key=lambda item: item[1]):
            f.write(str_makeup(lbl, 40) + str(coded) + '\n')
        f.close()
        print('>>>Done: Print data2vec in \'WordDic.txt\' & \'LabelDic.txt\'.')

    def _train_test_statistics(self,lbl,lbl_dic,mode):
        stt={}
        for i in range(len(lbl_dic)):
            stt[i]=0
        for i in lbl:
            stt[i]+=1
        if mode=='train':
            self.trainer_lbl_statistics=stt
        elif mode=='test':
            self.tester_lbl_statistics = stt
        else:
            print('>!>Internal error, train/test mode not specified.')

    def _batch_average(self,lbl,txt,L,R):
        l2,t2,L2,R2=list(lbl),list(txt),list(L),list(R)
        sl,sn=[],[]#label name,label count
        for l in lbl:
            if l not in sl:
                sl.append(l)
                sn.append(1)
            else:
                sn[sl.index(l)]+=1
        mx=max(sn)
        while min(sn)<mx:
            for i,v in enumerate(lbl):
                if sn[sl.index(v)]<mx:
                    l2.append(v)
                    t2.append(txt[i])
                    L2.append(L[i])
                    R2.append(R[i])
                    sn[sl.index(v)]+=1
            #print('test:',sn)
        return l2,t2,L2,R2

    def get_train_data(self,path):
        context, ori = []
        audio_data = path + "train_audio.mat"
        text_data = path + "test_audio.mat"

        scio.loadmat()
        return context, ori

    def get_test_data(self):
        return

    def _ramdon_select_from_trainertester(self,lbl, txt, al, ar, each_count=60):
        # 就是从train/test label 里面随机选一些出来
        dic = {}
        lbl_, txt_, al_, ar_ = [], [], [], []
        # for i,(l,t,L,R) in enumerate(zip(lbl,txt,al,ar)):
        for i, l in enumerate(lbl):
            #l = tuple(label)
            if l in dic:
                dic[l].append(i)
            else:
                dic[l] = [i]
        for l in dic.keys():
            indices = dic[l]
            random.shuffle(indices)
            for index in indices[:min(len(indices), each_count)]:
                lbl_.append(lbl[index])
                txt_.append(txt[index])
                al_.append(al[index])
                ar_.append(ar[index])
        return lbl_, txt_, al_, ar_


def main():

    path = r'D:/CNMC/hospital_data'
    #test_data.load()
    test_data=data(path=path)
    test_data.unclear_lbl.append('Monitor Vital Signs')
    test_data.diagnose=True
    test_data.auto_process(merge_unclear=True, load_audio=False)

    #multi-label
    if path[-1] != '\\':path += '\\'
    path = path.replace('\\', '/')
    mlb=open(path+'analyze/Multi-Labels.txt', 'w')
    mlb.write(str_makeup('XLSX NAME', 30) + 'NUM' + '  ' + str_makeup('GIVEN LABEL', 85) + str('SENTENCE' + '\n\n'))
    for key,val in test_data.data.items():
        if len(val[3])>1:
            mlb.write(str_makeup(key[0], 30) + str_makeup(key[1],3) + '  ' + str_makeup(str(val[3]), 85) + str(val[0]) + '\n')
    mlb.close()



    '''
    lbl, sent, L, R=test_data.get_trainer(average=True)
    lbl2, sent2, L2, R2=test_data.get_tester(average=True)

    print(lbl2.shape,sent2.shape,L2.shape,R2.shape)
    print(test_data.trainer_lbl_statistics)
    print(len(test_data.trainer_lbl_statistics))
    print(test_data.xlsx_list)
    '''
    '''
    #输出文件每一行都是一句话，所有话。必要的十个activity显示出来。其他隐藏。
    new_path=path+'TenActivity/'
    tar_list=['Back','GCS Calculation','Oxygen','Head','C-Spine','Pulse Check','Blood Pressure','Left lower extremity','Mouth','Abdomen']
    for key in sorted(test_data.data.keys()):
        val=test_data.data[key]
        lbl3,sent=val[3][0],val[0]
        f=open(new_path+key[0]+'.txt','a')
        if lbl3 in tar_list:
            wr=str_makeup(lbl3,20)+str(sent)
        else:
            wr=' '*20+str(sent)

        f.write(wr+'\n')
        f.close()
    print('>>>Done writing TenAct')
    '''
    # 统计lower10的数量
    tar_list = ['Back', 'GCS Calculation', 'Oxygen', 'Head', 'C-Spine', 'Pulse Check', 'Blood Pressure',
                        'Extremity', 'Mouth', 'Abdomen']
    counter = {}
    for label in tar_list:
        counter[label] = 0
    for value in test_data.data.values():
        if value[3][0] in tar_list:
            counter[value[3][0]] += 1
    print(counter)
    print("total = ", sum(counter.values()))
if __name__=='__main__':
    main()
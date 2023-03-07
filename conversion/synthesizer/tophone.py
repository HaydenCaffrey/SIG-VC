

with open('/NASdata/yangyg/code-switch-phone/MIXV2/DB4_training_data/DB4_phone.txt','r') as f, open('train.txt','r') as ff, open('DB4_rtvcmel.txt','w') as p:
    meta=f.readlines()
    train=ff.readlines()
    for idx in range(len(train)):
        phones=meta[idx]
        phlist=phones.strip().split('|')
        w1,w2,w3,w4,w5,w6=train[idx].strip().split('|')
        string=w1+'|'+w2+'|'+phlist[2]+'|'+w4+'|'+w5+'|'+w6
        p.write(string+'\n') 

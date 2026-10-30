
std_table = {
    'TTT':'F','TCT':'S','TAT':'Y','TGT':'C',
    'TTC':'F','TCC':'S','TAC':'Y','TGC':'C',
    'TTA':'L','TCA':'S','TAA':'X','TGA':'X',
    'TTG':'L','TCG':'S','TAG':'X','TGG':'W',

    'CTT':'L','CCT':'P','CAT':'H','CGT':'R',
    'CTC':'L','CCC':'P','CAC':'H','CGC':'R',
    'CTA':'L','CCA':'P','CAA':'Q','CGA':'R',
    'CTG':'L','CCG':'P','CAG':'Q','CGG':'R',

    'ATT':'I','ACT':'T','AAT':'N','AGT':'S',
    'ATC':'I','ACC':'T','AAC':'N','AGC':'S',
    'ATA':'I','ACA':'T','AAA':'K','AGA':'R',
    'ATG':'M','ACG':'T','AAG':'K','AGG':'R',

    'GTT':'V','GCT':'A','GAT':'D','GGT':'G',
    'GTC':'V','GCC':'A','GAC':'D','GGC':'G',
    'GTA':'V','GCA':'A','GAA':'E','GGA':'G',
    'GTG':'V','GCG':'A','GAG':'E','GGG':'G',
}

codon_61 = {
    'TTT':1,'TCT':2,'TAT':3,'TGT':4,
    'TTC':5,'TCC':6,'TAC':7,'TGC':8,
    'TTA':9,'TCA':10,'TTG':11,'TCG':12,
    'TGG':13,'CTT':14,'CCT':15,'CAT':16,
    'CGT':17,'CTC':18,'CCC':19,'CAC':20,
    'CGC':21,'CTA':22,'CCA':23,'CAA':24,
    'CGA':25,'CTG':26,'CCG':27,'CAG':28,
    'CGG':29,'ATT':30,'ACT':31,'AAT':32,
    'AGT':33,'ATC':34,'ACC':35,'AAC':36,
    'AGC':37,'ATA':38,'ACA':39,'AAA':40,
    'AGA':41,'ATG':42,'ACG':43,'AAG':44,
    'AGG':45,'GTT':46,'GCT':47,'GAT':48,
    'GGT':49,'GTC':50,'GCC':51,'GAC':52,
    'GGC':53,'GTA':54,'GCA':55,'GAA':56,
    'GGA':27,'GTG':58,'GCG':59,'GAG':60,
    'GGG':61,
}


# codon list in PAML mlc file
codonListY = [
    'TTT','TTC','TTA','TTG',
    'TCT','TCC','TCA','TCG',
    'TAT','TAC','TAA','TAG',
    'TGT','TGC','TGA','TGG',

    'CTT','CTC','CTA','CTG',
    'CCT','CCC','CCA','CCG',
    'CAT','CAC','CAA','CAG',
    'CGT','CGC','CGA','CGG',

    'ATT','ATC','ATA','ATG',
    'ACT','ACC','ACA','ACG',
    'AAT','AAC','AAA','AAG',
    'AGT','AGC','AGA','AGG',

    'GTT','GTC','GTA','GTG',
    'GCT','GCC','GCA','GCG',
    'GAT','GAC','GAA','GAG',
    'GGT','GGC','GGA','GGG',
]

aa_list = {
    'F': 1, 'S': 2, 'Y': 3, 'C': 4,
    'L': 5, 'X': 6, 'W': 7, 'P': 8,
    'H': 9, 'Q':10, 'R':11, 'I':12,
    'T':13, 'N':14, 'K':15, 'M':16,
    'V':17, 'A':18, 'D':19, 'G':20,
    'E':21,
}

nucleotide_list = {
    'A': 1,
    'G': 2,
    'C': 3, 
    'T': 4,
    
}

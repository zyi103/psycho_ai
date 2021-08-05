import sys
print(sys.path)

import two_afc

embedding = two_afc.get_glove_100d()
target_occupations = ['accountant', 'supervisor', 'worker', 'clerk', 'instructor', 'inspector',
          'electrician','appraiser', 'administrator', 'receptionist', 'advisor', 'chemist',
          'planner','paralegal', 'veterinarian', 'psychologist',  'baker', 'teacher', 
          'lawyer','nutritionist', 'hairdresser','pathologist', 'surgeon', 'practitioner', 
          'carpenter']
female_male_pairs = [['woman', 'man'],
                     ['female', 'male'],
                     ['she', 'he'],
                     ['her', 'him'],
                     ['hers', 'his'],
                     ['daughter', 'son'],
                     ['girl', 'boy'],
                     ['sister', 'brother']]
                
pse_score = two_afc.pse(embedding, target_occupations, female_male_pairs)
print(pse_score)
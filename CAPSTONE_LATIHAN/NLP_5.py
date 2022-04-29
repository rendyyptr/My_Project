import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
data = 'In the town of Athy one Jeremy Lanigan \n ' \
       'Battered away til he hadnt a pound. \n' \
       'His father died and made him a man again \n ' \
       'Left him a farm and ten acres of ground. \n' \
       'He gave a grand party for friends and relations \n' \
       'Who didnt forget him when come to the wall, \n' \
       'And if youll but listen Ill make your eyes glisten \n' \
       'Of the rows and the ructions of Lanigans Ball. \n' \
       'Myself to be sure got free invitation, \n' \
       'For all the nice girls and boys I might ask, \n' \
       'And just in a minute both friends and relations \n' \
       'Were dancing round merry as bees round a cask. \n' \
       'Judy ODaly, that nice little milliner, \n' \
       'She tipped me a wink for to give her a call, \n' \
       'And I soon arrived with Peggy McGilligan \n' \
       'Just in time for Lanigans Ball. \n' \
       'There were lashings of punch and wine for the ladies, \n' \
       'Potatoes and cakes; there was bacon and tea, \n' \
       'There were the Nolans, Dolans, OGradys \n' \
       'Courting the girls and dancing away. \n' \
       'Songs they went round as plenty as water, \n' \
       'The harp that once sounded in Taras old hall,\n' \
       'Sweet Nelly Gray and The Rat Catchers Daughter,\n' \
       'All singing together at Lanigans Ball. \n' \
       'They were doing all kinds of nonsensical polkas \n' \
       'All round the room in a whirligig. \n' \
       'Julia and I, we banished their nonsense \n' \
       'And tipped them the twist of a reel and a jig. \n' \
       'Ach mavrone, how the girls got all mad at me \n' \
       'Danced til youd think the ceiling would fall. \n' \
       'For I spent three weeks at Brooks Academy \n' \
       'Learning new steps for Lanigans Ball. \n' \
       'Three long weeks I spent up in Dublin, \n' \
       'Three long weeks to learn nothing at all,\n ' \
       'Three long weeks I spent up in Dublin, \n' \
       'Learning new steps for Lanigans Ball. \n' \
       'She stepped out and I stepped in again, \n' \
       'I stepped out and she stepped in again, \n' \
       'She stepped out and I stepped in again, \n' \
       'Learning new steps for Lanigans Ball. \n' \
       'Boys were all merry and the girls they were hearty \n' \
       'And danced all around in couples and groups, \n' \
       'Til an accident happened, young Terrance McCarthy \n' \
       'Put his right leg through miss Finnertys hoops. \n' \
       'Poor creature fainted and cried Meelia murther, \n' \
       'Called for her brothers and gathered them all. \n' \
       'Carmody swore that hed go no further \n' \
       'Til he had satisfaction at Lanigans Ball. \n' \
       'In the midst of the row miss Kerrigan fainted, \n' \
       'Her cheeks at the same time as red as a rose. \n' \
       'Some of the lads declared she was painted, \n' \
       'She took a small drop too much, I suppose. \n' \
       'Her sweetheart, Ned Morgan, so powerful and able, \n' \
       'When he saw his fair colleen stretched out by the wall, \n' \
       'Tore the left leg from under the table \n' \
       'And smashed all the Chaneys at Lanigans Ball. \n' \
       'Boys, oh boys, twas then there were runctions. \n' \
       'Myself got a lick from big Phelim McHugh. \n' \
       'I soon replied to his introduction \n' \
       'And kicked up a terrible hullabaloo. \n' \
       'Old Casey, the piper, was near being strangled. \n' \
       'They squeezed up his pipes, bellows, chanters and all. \n' \
       'The girls, in their ribbons, they got all entangled \n' \
       'And that put an end to Lanigans Ball.'

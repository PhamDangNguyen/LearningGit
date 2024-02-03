import os
import re
import json
import random
import logging
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)
SAMPLERATE = 16000
NUMBER_UTT = 5531


def prepare_data(
    data_original,
    path_txt_info,
    save_json_train,
    save_json_valid,
    save_json_test,
    split_ratio=[80, 10, 10],
    different_speakers=False,
    test_spk_id=1,
    seed=12,
):
    """
    Prepares the json files for the IEMOCAP dataset.

    Arguments
    ---------
    data_original : str
        Path to the folder where the original IEMOCAP dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    split_ratio: list
        List composed of three integers that sets split ratios for train,
        valid, and test sets, respecively.
        For instance split_ratio=[80, 10, 10] will assign 80% of the sentences
        to training, 10% for validation, and 10% for test.
    test_spk_id: int
        Id of speaker used for test set, 10 speakers in total.
        Here a leave-two-speaker strategy is used for the split,
        if one test_spk_id is selected for test, the other spk_id in the same
        session is automatically used for validation.
        To perform a 10-fold cross-validation,
        10 experiments with test_spk_id from 1 to 10 should be done.
    seed : int
        Seed for reproducibility

    Example
    -------
    >>> data_original = '/path/to/iemocap/IEMOCAP_full_release'
    >>> prepare_data(data_original, 'train.json', 'valid.json', 'test.json')
    """
    # setting seeds for reproducible code.
    random.seed(seed)

    info_dict = transform_data(data_original,path_txt_info)#tao tu dien info

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )

    data_split = split_sets(info_dict, split_ratio)#cat tu dien thanh 3 tap

    # Creating json files
    create_json(data_split["train"], save_json_train)
    create_json(data_split["valid"], save_json_valid)
    create_json(data_split["test"], save_json_test)


def create_json(wav_list, json_file_save):
    """
    Creates the json file given a list of wav information.

    Arguments
    ---------
    wav_list : list of list
        The list of wav information (path, label, sex).
    json_file : str
        The path of the output json file (train, valid , test)
    """

    

    json_dict = {}
    for obj in wav_list:
        wav_file = obj[0]
        regions = obj[1]
        # Read the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        uttid = wav_file.split("/")[-1][:-4]

        # Create entry for this utterance
        json_dict[uttid] = {
            "wav": wav_file,
            "length": duration,
            "regions": regions,
        }
        # print(json_dict[uttid])
    # Writing the dictionary to the json file
    with open(json_file_save, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file_save} successfully created!")


def skip(*filenames):
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True

def split_sets(data_dict, split_ratio):
    """Randomly splits the wav list into training, validation, and test lists.
    Note that a better approach is to make sure that all the classes have the
    same proportion of samples (e.g, spk01 should have 80% of samples in
    training, 10% validation, 10% test, the same for speaker2 etc.). This
    is the approach followed in some recipes such as the Voxceleb one. For
    simplicity, we here simply split the full list without necessarly
    respecting the split ratio within each class.

    Arguments
    ---------
    speaker_dict : list
        a dictionary of speaker id and its corresponding audio information
    split_ratio: list
        List composed of three integers that sets split ratios for train,
        valid, and test sets, respectively.
        For instance split_ratio=[80, 10, 10] will assign 80% of the sentences
        to training, 10% for validation, and 10% for test.

    Returns
    ------
    dictionary split containing train, valid, and test splits.
    """

    wav_list = []
    for key in data_dict.keys():
        wav_list.append(data_dict[key][0])
        # print(info[key][0][0])

    # Random shuffle of the list
    random.shuffle(wav_list)
    tot_split = sum(split_ratio)
    tot_snts = len(wav_list)
    data_split = {}
    splits = ["train", "valid"]

    for i, split in enumerate(splits):
        n_snts = int(tot_snts * split_ratio[i] / tot_split)
        data_split[split] = wav_list[0:n_snts]
        del wav_list[0:n_snts]
    data_split["test"] = wav_list

    return data_split




def load_in4(path_txt_info):
    """

    Arguments
    ---------
        pathSession: str
            Path folder of text info speaker-metadata.tsv.
    Returns
    -------
        improvisedUtteranceList: list
            List includes [Id_speaker, region, Sex].
    """

    List_in4 = []

    
    with open(path_txt_info, 'r') as f:
        for i in f:
            text_in4 = i.strip().split()
            List_in4.append([text_in4[0],text_in4[1],text_in4[2]])
    return List_in4

def transform_data(path_to_dataWav,path_txt_info):
    """
    Arguments
    ---------
        pathSession: str
            Path folder's data wave
            Path info meta.tsv
    Returns
    -------
        improvisedUtteranceList: list
           Array [Path_wav, region, Sex].
           data_dict
    """
    # id_speaker, regions, Sex = load_in4(path_txt_info)
    infor_wav = []

    for id_speaker, regions, Sex in load_in4(path_txt_info):
        for i in os.listdir(path_to_dataWav):
            if str(i) == str(id_speaker):
                for wav_name in os.listdir(os.path.join(path_to_dataWav,id_speaker)):
                    path_wav = os.path.join(os.path.join(path_to_dataWav,id_speaker),wav_name)
                    infor_wav.append([path_wav.replace("\\","/"),regions,Sex])

    data_dict = {}

    for idx in range(len(infor_wav)):
        if idx not in data_dict:
            data_dict[idx] = []
        data_dict[idx].append(infor_wav[idx])

    return data_dict
    


if __name__ == "__main__":
    path_txt_info = "C:/Users/dangn/OneDrive/Máy tính/VoicePytorch/vietnam_celeb_part_data/speaker-metadata.tsv"
    path_to_dataWav = "C:/Users/dangn/OneDrive/Máy tính/VoicePytorch/vietnam_celeb_part_data/data"
    # load_in4(path_txt_info)
    # info = transform_data(path_to_dataWav=path_to_dataWav,path_txt_info=path_txt_info)
    # for key in info.keys():
    #     print(info[key][0])

    # data_split_info = split_sets(info,[80,10,10])
    # for i in data_split_info["train"]:
    #     print(i)
    # print(len(data_split_info["train"]))
    # print(len(data_split_info["test"]))
    # print(len(data_split_info["valid"]))
    # create_json(data_split_info["test"],json_file_save)
    
    json_file_save_train = "C:/Users/dangn/OneDrive/Máy tính/VoicePytorch/Wave2vec_dialect_regions/data/json_file/train.json"
    json_file_save_test = "C:/Users/dangn/OneDrive/Máy tính/VoicePytorch/Wave2vec_dialect_regions/data/json_file/test.json"
    json_file_save_valid = "C:/Users/dangn/OneDrive/Máy tính/VoicePytorch/Wave2vec_dialect_regions/data/json_file/valid.json"
     
    prepare_data(path_to_dataWav,path_txt_info,json_file_save_train,json_file_save_valid,json_file_save_test)
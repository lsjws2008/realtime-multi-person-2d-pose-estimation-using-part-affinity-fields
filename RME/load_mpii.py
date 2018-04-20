from scipy.io import loadmat
def generate_file():
    data = loadmat('mpii_human_pose_v1_u12_1.mat')
    imgs = data['RELEASE'][0][0][0][0]
    """
    img_num = 4
    one_img_file_name = imgs[img_num][0][0]
    one_img_vadio_ind = imgs[img_num][3][0]
    one_image_persons = imgs[img_num][1][0]
    one_image_one_person_annos = one_image_persons[0][4][0][0][0][0]
    anno_ind = 0
    one_anno_x = 0
    one_anno_y = 1
    one_anno_id = 2
    one_anno_visible = 3
    one_image_one_person_one_anno_x = one_image_one_person_annos[anno_ind][one_anno_x][0][0]
    """
    all_data = []
    noob_ind = [6054, 7913, 8104, 10799, 24730]
    for i, img in enumerate(imgs):
        if i in noob_ind:
            continue
        one_img_data = {}
        one_img_data['filename'] = img[0][0][0][0][0]
        viind = img[3][0]
        if len(viind) == 0:
            one_img_data['viind'] = 0
            one_img_data['seq'] = 0
        else:
            one_img_data['viind'] = viind[0]
            one_img_data['seq'] = img[2][0][0]
        if len(img[1]) != 0:
            if img[1][0][0] == None:
                one_img_data['persons_anno'] = 0
            elif len(img[1][0][0]) < 7 :
                #the last 0 means the first person in this img.
                one_img_data['persons_anno'] = 0
            else :
                persons = []
                for person in img[1][0]:
                    if len(person[4]) != 0:
                        one_person = []
                        for anno in person[4][0][0][0][0]:
                            if len(anno[3]) != 0:
                                if anno[3][0][0] == 1:
                                    one_person.append(anno[0][0][0])
                                    one_person.append(anno[1][0][0])
                                    one_person.append(anno[2][0][0])
                        persons.append(one_person)
                one_img_data['persons_anno'] = persons
        else:
            one_img_data['persons_anno'] = 0
        all_data.append(one_img_data)
    return all_data

if __name__ == '__main__':
    data = generate_file()
    for i in data:
        print(i['persons_anno'])
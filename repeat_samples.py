def repeat(srcs, convs, tags):
    new_src = []
    new_conv = []
    new_tag = []
    for i in zip(srcs, convs, tags):
        tag_list = i[2].split(";")
        for j in range(len(tag_list)):
            new_src.append(i[0])
            new_conv.append(i[1])
        new_tag += tag_list
    assert len(new_conv) == len(new_src) == len(new_tag)
    return new_src, new_conv, new_tag


def write_to_file(file_path, entities):
    f = open(file_path, "w", encoding='utf-8')
    for t in entities:
        f.write(t)
        f.write("\n")
    f.close()


def read_file(file_path):
    f = open(file_path, "r", encoding='utf-8')
    lines = f.readlines()
    f.close()
    return [l.rstrip("\n") for l in lines]


if __name__ == '__main__':
    key = "train"
    src_path = "./data/" + key + "_post.txt"
    conv_path = "./data/" + key + "_conv.txt"
    tag_path = "./data/" + key + "_tag.txt"
    srcs = read_file(src_path)
    convs = read_file(conv_path)
    tags = read_file(tag_path)
    new_data = repeat(srcs, convs, tags)
    write_to_file('./data/new_'+key+ "_post.txt", new_data[0])
    write_to_file('./data/new_' + key + "_conv.txt", new_data[1])
    write_to_file('./data/new_' + key + "_tag.txt", new_data[2])



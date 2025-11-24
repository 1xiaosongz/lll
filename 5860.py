import json
import math
from itertools import chain
import prediction
from matplotlib import pyplot as plt
import numpy as np


def read_json_txt(file_path):
    """读取 JSON 格式的文本文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 解析 JSON 数据
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    return None

def find_peaks(file_path):
    # 读取文件并解析数据
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # 尝试反序列化每行数据（支持多种分隔符）
            values = line.replace(',', ' ').split()
            for val in values:
                try:
                    num = float(val)
                    data.append(num)
                except ValueError:
                    continue

    if not data:
        print("警告：未找到有效数值数据")
        return []

    # 计算平均值
    avg = sum(data) / len(data)
    threshold = avg * 10  # 超过平均值30%的阈值
    # 筛选峰值
    peaks = [val for val in data if val >= threshold]

    # 打印结果
    print(f"数据总量: {len(data)}")
    print(f"平均值: {avg:.4f}")
    print(f"峰值阈值: {threshold:.4f} (平均值的130%)")
    print(f"找到峰值数量: {len(peaks)}")

    return peaks

def Calculate_difference(arr,arr1):
    C = []  # 存储满足条件的 x
    i, j = 0, 0  # 初始化索引
    # 循环直到任一索引超出数组范围
    while i < len(arr) and j < len(arr1):
        x = arr[i] - arr1[j]
        # 如果 |x| < 500，则将 x 存入 C
        if abs(x) < 500:
            C.append(x)
            i += 1  # a 索引增加
        else:
            j += 1  # b 索引增加
    difference = max(C)
    print(difference)
    return difference


def group_and_count_2d_arrays(pairs):
    # 初始化结果字典
    total_counts = {
        "小于50": 0,
        "50-100": 0,
        "100-150": 0,
        "150-200": 0,
        "大于200": 0
    }
    grouped_x_values = {
        "小于50": [],
        "50-100": [],
        "100-150": [],
        "150-200": [],
        "大于200": []
    }
    # 遍历所有元组
    for pair in pairs:
        x, y = pair
        diff = abs(x - y)

        # 确定所属区间
        if diff < 50:
            total_counts["小于50"] += 1
            grouped_x_values["小于50"].append(y)
        elif 50 <= diff < 100:
            total_counts["50-100"] += 1
            grouped_x_values["50-100"].append(y)
        elif 100 <= diff < 150:
            total_counts["100-150"] += 1
            grouped_x_values["100-150"].append(y)
        elif 150 <= diff < 200:
            total_counts["150-200"] += 1
            grouped_x_values["150-200"].append(y)
        else:  # diff >= 200
            total_counts["大于200"] += 1
            grouped_x_values["大于200"].append(y)

    return total_counts, grouped_x_values


def group_array(arr,file_size, group_size=3200):#6S  的改成6400
    if not arr:
        return [[] for _ in range(file_size)]

        # 初始化所有组为空列表
    groups = [[] for _ in range(file_size)]

    # 将元素分配到对应的组并进行调整
    for num in arr:
        group_index = int(num // group_size)
        # 确保索引在范围内
        if 0 <= group_index < file_size:
            adjusted_value = num - group_size * group_index
            groups[group_index].append(adjusted_value)
    return groups

def group_and_count_d_arrays(array):
    # 检查输入是否是一维数组
    if not isinstance(array[0], list):
        # 如果是一维数组，将其转换为二维数组（只有一个子数组）
        array = [array]

    # 初始化总计数器
    total_counts = {
        "小于50": 0,
        "50-100": 0,
        "100-150": 0,
        "150-200": 0,
        "大于200": 0
    }
    total_elements = 0
    # 处理每个子数组
    for arr in array:
        # 统计当前子数组的各组数量
        for num in arr:
            if num < 50:
                total_counts["小于50"] += 1
            elif 50 <= num < 100:
                total_counts["50-100"] += 1
            elif 100 <= num < 150:
                total_counts["100-150"] += 1
            elif 150 <= num <= 200:
                total_counts["150-200"] += 1
            else:  # num > 200
                total_counts["大于200"] += 1
        # 更新总元素数
        total_elements += len(arr)

    # 计算总体百分比（带百分号）
    if total_elements > 0:
        total_percentages = {
            group: f"{(count / total_elements) * 100:.2f}%"
            for group, count in total_counts.items()
        }
    else:
        total_percentages = {group: "0.00%" for group in total_counts.keys()}

    # 返回结果
    return total_counts, total_percentages

def calculate_average_difference(data_list):
    differences = []
    for item in data_list:
        x, y = item
        differences.append(y - x)
    # 计算平均值
    average = int(sum(differences) / len(differences))
    return average


def normalize_list(arr, method='minmax'):
    """
    列表归一化函数
    :param arr: 输入列表（支持大数值）
    :param method: 归一化方式，可选'minmax'或'zscore'
    :return: 归一化后的新列表
    """
    if not arr:
        return []

    n = len(arr)

    # Min-Max归一化 [0,1]
    if method == 'minmax':
        min_val = min(arr)
        max_val = max(arr)
        if min_val == max_val:
            return [0.0] * n
        return [(x - min_val) / (max_val - min_val) for x in arr]

    # Z-score标准化 均值为0，标准差为1
    elif method == 'zscore':
        mean = sum(arr) / n
        # 计算标准差（分母为n的贝塞尔校正）
        variance = sum((x - mean) ** 2 for x in arr) / n
        std = variance ** 0.5
        if std == 0:
            return [0.0] * n
        return [(x - mean) / std for x in arr]

    else:
        raise ValueError("不支持的归一化方式，请选择'minmax'或'zscore'")

def Manual_RT_screening(numeric_data):
    # 把手动标点分开R T
    numeric_data=normalize_list(numeric_data)
    if numeric_data:
        avg = sum(numeric_data) / len(numeric_data)
    else:
        avg = 0  # 处理空列表情况
        print("警告：未找到有效数值数据")

    pos =0.9 # 超过平均值30%的阈值
    peaks =[]
    for i,_ in enumerate(numeric_data):
        if numeric_data[i] > avg*(1+pos) or numeric_data[i] < avg*(1-pos):
            peaks.append((i,numeric_data[i]))

    #把peak的检测位置左右扩n个点
    leftPeak=[(index , value) for index, value in peaks]   #-5
    rightPeak=[(index , value) for index, value in peaks]   #+5
    # print(peaks,"   +5     peaks          ")
    n = len(ecg_II)
    seen = {}  # 用于去重的字典
    # 使用chain高效连接三个数组，避免创建大临时数组
    all_peaks = chain(peaks, leftPeak, rightPeak)

    for index, value in all_peaks:
        # 检查索引是否在有效范围内
        if 0 <= index < n:
            # 去重：保留第一个出现的值（如果想要保留最后一个，去掉if判断）
            if index not in seen:
                seen[index] = (index, value)
    # 转换为列表并排序
    peaks = sorted(seen.values(), key=lambda x: x[0])
    avg=math.floor(np.mean(ecg_II))
    #删除临近的index
    flag=0
    Tmp=()
    rtIndex=[]
    for i in peaks:
        if flag==0:
            flag=i[0]
            Tmp=(i[0],abs(ecg_II[i[0]]-avg))
        elif flag!=0 and i[0]-flag<60:
             if abs(ecg_II[i[0]]-avg)>Tmp[1]:
                 Tmp=(i[0],abs(ecg_II[i[0]]-avg))
        elif i[0]-flag >60:
            rtIndex.append(Tmp[0])
            Tmp = (i[0], abs(ecg_II[i[0]] - avg))
            flag=i[0]
    if len(peaks)==0:
        print("no peaks detected")
        return [],[]
    rtIndex.append(Tmp[0])
    # print(rtIndex)
    #求平均偏离
    sumBios =[]
    for i in rtIndex:
        sumBios.append(abs(ecg_II[i]-avg))
    Bios = math.floor(sum(sumBios) / len(sumBios))
    # print("Bios:",Bios)
    R = []
    T = []
    for i in rtIndex:
        if ecg_II[i] > avg:
            if ecg_II[i] - avg >= 0.9 * Bios:
                R.append(i)
            else:
                T.append(i)
        else:
            if abs(ecg_II[i] - avg) >= 0.5 * Bios:
                R.append(i)
            else:
                T.append(i)
    print("=====================================")
    return R,T


def count_elements(err_c_r,i):
    if not err_c_r:
        a = 0
        b = []
    else:
        a = len(err_c_r[i])
        b = err_c_r[i]
    return a,b
def error_detection(arr,arr1):
    matching_T = [pair[1] for pair in arr]
    # 找出 T 中不在 matching_T 中的值
    missing_values = [t for t in arr1 if t not in matching_T]
    return missing_values
def missed_detection(arr,arr1):
    matching_T = [pair[0] for pair in arr]
    # 找出 T 中不在 matching_T 中的值
    missing_values = [t for t in arr1 if t not in matching_T]
    return missing_values

def process_and_combine_arrays(array_of_arrays):
    """
    处理多个数组并合并元素

    参数:
    array_of_arrays: 包含多个数组的数组（列表的列表）

    返回:
    合并后的数组，包含所有处理后的元素
    """
    combined_array = []
    for i, arr in enumerate(array_of_arrays):
        # 计算当前数组的偏移量
        offset = 3200 * i                           #6S改成  6400
        # 处理当前数组的元素并添加到合并数组
        for element in arr:
            combined_array.append(element + offset)
    return combined_array
def grouping(data):
    max_x = max(max(pair) for pair in data)
    max_group = max_x // 3200
    # 初始化分组列表
    groups = [[] for _ in range(max_group + 1)]

    # 将数据分配到各组
    for pair in data:
        x, y = pair
        group_index = x // 3200
        groups[group_index].append(pair)
    # 对每组应用相应的减法操作
    result_groups = []
    for i, group in enumerate(groups):
        if i == 0:
            # 第0组保持不变
            result_groups.append(group)
        else:
            # 其他组减去6400*i
            adjusted_group = []
            for (x, y) in group:
                adjusted_x = x - 3200 * i
                adjusted_y = y - 3200 * i
                adjusted_group.append((adjusted_x, adjusted_y))
            result_groups.append(adjusted_group)
    return result_groups


def remove_elements_with_small_diff(arr):
    """
    删除数组中相邻元素差值小于10的前一个元素（改进版本）

    参数:
    arr: 输入数组

    返回:
    处理后的新数组
    """
    if not arr or len(arr) < 2:
        return arr.copy()

    result = []
    i = 0

    while i < len(arr):
        # 如果是最后一个元素，直接添加
        if i == len(arr) - 1:
            result.append(arr[i])
            break

        # 检查当前元素和下一个元素的差值
        if abs(arr[i] - arr[i + 1]) < 30:
            # 差值小于10，跳过当前元素
            i += 1
        else:
            # 差值大于等于10，保留当前元素
            result.append(arr[i])
            i += 1

    return result


def remove_elements_recursive_with_deleted(arr, deleted=None):
    """
    递归版本，同时记录被删除的元素
    """
    if deleted is None:
        deleted = []

    for i in range(len(arr) - 1):
        if arr[i + 1] - arr[i] <= 500:
            # 记录被删除的元素
            deleted.append(arr[i + 1])
            # 删除后一个元素并递归调用
            new_arr = arr[:i + 1] + arr[i + 2:]
            return remove_elements_recursive_with_deleted(new_arr, deleted)
    return arr, deleted
def difference_value(R):
    R_pairs_with_small_diff = []
    for i in range(len(R)):
        for j in range(i + 1, len(R)):
            if abs(R[i] - R[j]) < 200:
                R_pairs_with_small_diff.append((R[i], R[j]))
    return R_pairs_with_small_diff
def multi(R_pairs_with_small_diff,d_R_T_set):
    R_in_d_R_T=[]
    for pair in R_pairs_with_small_diff:
        for val in pair:
            if val in d_R_T_set and val not in R_in_d_R_T:
                R_in_d_R_T.append(val)
    return R_in_d_R_T

def simple_extract(arr):
    """
    简化版本：执行两次差值计算并提取元素
    """

    def one_pass(current_arr):
        extracted = []
        i = 0
        while i < len(current_arr) - 1:
            if 800 < current_arr[i + 1] - current_arr[i] < 900:
                extracted.append(current_arr[i + 1])
                current_arr.pop(i + 1)
            else:
                i += 1
        return extracted

    remaining = arr.copy()
    first_extracted = one_pass(remaining)
    second_extracted = one_pass(remaining)

    return remaining, first_extracted + second_extracted


#//////////////////////////////////

def simple_extract_1(arr):
    """
    简化版本：执行两次差值计算并提取元素
    """

    def one_pass(current_arr):
        extracted = []
        i = 0
        while i < len(current_arr) - 1:
            if current_arr[i + 1] - current_arr[i] < 400:
                extracted.append(current_arr[i + 1])
                current_arr.pop(i + 1)
            else:
                i += 1
        return extracted

    remaining = arr.copy()
    first_extracted = one_pass(remaining)
    second_extracted = one_pass(remaining)

    return remaining, first_extracted + second_extracted

def simple_extract_3(arr):
    """
    简化版本：执行两次差值计算并提取元素
    """

    def one_pass(current_arr):
        extracted = []
        i = 0
        while i < len(current_arr) - 1:
            if current_arr[i + 1] - current_arr[i] < 300:
                extracted.append(current_arr[i + 1])
                current_arr.pop(i + 1)
            else:
                i += 1
        return extracted

    remaining = arr.copy()
    first_extracted = one_pass(remaining)
    second_extracted = one_pass(remaining)

    return remaining, first_extracted + second_extracted


def simple_extract_2(arr):
    """
    简化版本：执行两次差值计算并提取元素
    """

    def one_pass(current_arr):
        extracted = []
        i = 0
        while i < len(current_arr) - 1:
            if  current_arr[i + 1] - current_arr[i] < 430:
                extracted.append(current_arr[i + 1])
                current_arr.pop(i + 1)
            else:
                i += 1
        return extracted

    remaining = arr.copy()
    first_extracted = one_pass(remaining)
    second_extracted = one_pass(remaining)

    return remaining, first_extracted + second_extracted

def simple_extract_4(arr):
    """
    简化版本：执行两次差值计算并提取元素
    """

    def one_pass(current_arr):
        extracted = []
        i = 0
        while i < len(current_arr) - 1:
            if current_arr[i + 1] - current_arr[i] < 330:
                extracted.append(current_arr[i + 1])
                current_arr.pop(i + 1)
            else:
                i += 1
        return extracted

    remaining = arr.copy()
    first_extracted = one_pass(remaining)
    second_extracted = one_pass(remaining)

    return remaining, first_extracted + second_extracted
# 使用示例
if __name__ == "__main__":
    import os
    base_path = "20/3015860-0013/"
    # 获取 base_path 下的所有文件名
    file_names = os.listdir(base_path)
    # 拼接完整路径
    file_paths = [os.path.join(base_path, name) for name in file_names]
    x=-1
    lowerR=[]
    lowerT=[]
    ComputerR = []
    ComputerT = []
    dR=[]
    dT=[]
    a1=[]
    R_dot=[]
    T_dot=[]
    RTIL=[]
    MRTIL = []
    CRTIL = []
    ecg=[]
    ecg_size = 0
    fil_size = 0
    c=[]
    for path in file_paths:
        fil_size=fil_size+1
        file_path_root = path
        # print(file_path_root," 222")
        file_path = file_path_root+"/rawEcgData.txt"  # 可以是任何扩展名，重点是内容格式
        json_data = read_json_txt(file_path)
        res = json_data['rawEcgData']
        ecg_II=[]
        for v in res:
            ecg_II.extend(v[1])
        if ecg_size==0:
            ecg_size = len(ecg_II)
        ecg.extend(ecg_II)
        c.append(res)

        file_path1 = file_path_root+"/rawBloodData.txt"  # 可以是任何扩展名，重点是内容格式
        json_data1 = read_json_txt(file_path1)
        numeric_data = json_data1['rawBloodData']

        file_path2 = file_path_root + "/R_DetectedByComputer.txt"  # 可以是任何扩展名，重点是内容格式
        json_data2 = read_json_txt(file_path2)
        R_D_C = json_data2['R_DetectedByComputer']
        ComputerR.append(R_D_C[0])

        file_path3 = file_path_root + "/R_DetectedByDowner.txt"  # 可以是任何扩展名，重点是内容格式
        json_data3 = read_json_txt(file_path3)
        R_D_D = json_data3['R_DetectedByDowner']
        lowerR.append(R_D_D[0])

        file_path4 = file_path_root + "/T_DetectedByComputer.txt"  # 可以是任何扩展名，重点是内容格式
        json_data4 = read_json_txt(file_path4)
        T_D_C = json_data4['T_DetectedByComputer']
        ComputerT.append(T_D_C[0])

        file_path5 = file_path_root + "/T_DetectedByDowner.txt"  # 可以是任何扩展名，重点是内容格式
        json_data5 = read_json_txt(file_path5)
        T_D_D = json_data5['T_DetectedByDowner']
        lowerT.append(T_D_D[0])

        file_path7 = file_path_root + "/unusedRT.txt"  # 可以是任何扩展名，重点是内容格式

        if os.path.exists(file_path7):
            json_data7 = read_json_txt(file_path7)
            U_R = json_data7['unusedRT'][0]
            U_T = json_data7['unusedRT'][1]
            dR.append(U_R)
            dT.append(U_T)
        else:
            U_R = []
            U_T = []
            dR.append(U_R)
            dT.append(U_T)

        file_path6 = file_path_root + "/RTInterval.txt"  # 可以是任何扩展名，重点是内容格式

        #如果文件没有下发间隔手动添加为 -1
        if not os.path.exists(file_path6):
            RTI = -1
            RTIL.append(-1)
        else:
            json_data6 = read_json_txt(file_path6)
            RTI = json_data6['RTInterval']
            RTIL.append(RTI)


        R,T=Manual_RT_screening(numeric_data)    #numeric_data保存的手动标点
        print(R)
        print(T)
        print(file_path_root)
        # ///////////////////////
        # plt.vlines(x=R, ymin=min(ecg_II), ymax=max(ecg_II), colors='r', linestyles='solid', linewidths=1)
        # plt.vlines(x=T, ymin=min(ecg_II), ymax=max(ecg_II), colors='g', linestyles='solid', linewidths=1)
        # # # plt.vlines(x=R_D_C,  ymin=min(ecg_II), ymax=max(ecg_II), colors='g', linestyles='solid', linewidths=1)
        # # # plt.vlines(x=R_D_D,  ymin=min(ecg_II), ymax=max(ecg_II), colors='b', linestyles='solid', linewidths=1)
        # # # # plt.vlines(x=T_D_C,  ymin=min(ecg_II), ymax=max(ecg_II), colors='c', linestyles='solid', linewidths=1)#蓝
        # # # plt.vlines(x=T_D_D,  ymin=min(ecg_II), ymax=max(ecg_II), colors='m', linestyles='solid', linewidths=1)#粉m
        # # #
        # # # # # 绘制 ECG 信号（后画，确保不被竖线遮挡）
        # plt.title(file_path_root)
        # plt.plot(ecg_II, 'k-', label='ECG II')
        # plt.show()
        # plt.close()

        # /////////////////
        R_dot.append(R)
        T_dot.append(T)
        # 手动标点匹配上的RT,用于计算手动标记的平均间隔
        Num  =prediction.find_mutual_nearest_rt(R, T)
        ManualInterval=calculate_average_difference(Num)

        MRTIL.append(ManualInterval)
        # 下位机标点匹配上的RT,用于计算手动标记的平均间隔
        if len(R_D_D[0]) >0 and len(T_D_D[0])>0:
            Num_1 = prediction.find_mutual_nearest_rt(R_D_D[0], T_D_D[0])
            lowerComputer = calculate_average_difference(Num_1)
        else:
            Num_1 = 0
            lowerComputer = -2
        CRTIL.append(lowerComputer)
        file_path11 = file_path_root + "/interior_Interval.txt"
        date2 = {
            "Manual_Interval":ManualInterval,    #手动标点的间隔
            "Downer_Interval":lowerComputer,   #实际存下来的数据的间隔
            "Computer_Interval":RTI ,         #上位机下发的间隔
            "Computer_R_Num":len(R_D_C[0]),   #上位机R标点个数
            "Downer_R_Num":len(R_D_D[0]),     #下位机R点个数
            "Computer_T_Num": len(T_D_C[0]),    #上位机T标点个数
            "Downer_T_Num": len(T_D_D[0]),       #下位机T点个数
        }
        with open(file_path11, 'w',encoding='utf-8') as file:
            json.dump(date2, file,indent=4,ensure_ascii=False)
            file.write('\n')

        plt.show()
        plt.close()
        if json_data1 is not None:
            print("成功读取 JSON 数据:")
            # print(peaks)

    ComputerR=process_and_combine_arrays(ComputerR)
    lowerR=process_and_combine_arrays(lowerR)
    ComputerT=process_and_combine_arrays(ComputerT)
    lowerT=process_and_combine_arrays(lowerT)
    R_dot_1=process_and_combine_arrays(R_dot)
    T_dot_1=process_and_combine_arrays(T_dot)

    R_dot_2 = remove_elements_with_small_diff(R_dot_1)           #删除距离很近的点
    T_dot_2 = remove_elements_with_small_diff(T_dot_1)

    # T_dot_4,issues_R = simple_extract_1(T_dot_2)
    # R_dot_4 = R_dot_2 + issues_R
    # R_dot_4.sort()
    # #
    R_dot_3,issues_T = simple_extract_3(R_dot_2)
    T_dot_3 = T_dot_2 + issues_T
    T_dot_3.sort()

    # # #
    T_dot_6, issues_R_1 = simple_extract_1(T_dot_3)
    R_dot_6 = R_dot_3 + issues_R_1
    R_dot_6.sort()
    # # #
    R_dot_5, issues_T_1 = simple_extract_3(R_dot_6)
    T_dot_5 = T_dot_6 + issues_T_1
    T_dot_5.sort()

    R_dot_10, issues_T_2 = simple_extract_1(R_dot_5)
    T_dot_10 = T_dot_5 + issues_T_2
    T_dot_10.sort()
    # #

    T_dot_9, issues_R_3 = simple_extract_1(T_dot_10)
    R_dot_9 = R_dot_10 + issues_R_3
    R_dot_9.sort()

    R_dot_10, issues_T_2 = simple_extract_1(R_dot_9)
    T_dot_10 = T_dot_9 + issues_T_2
    T_dot_10.sort()

    T_dot_11, issues_R_3 = simple_extract_1(T_dot_10)
    R_dot_11 = R_dot_10 + issues_R_3
    R_dot_11.sort()

    R_dot_12, issues_T_2 = simple_extract_3(R_dot_11)
    T_dot_12 = T_dot_11 + issues_T_2
    T_dot_12.sort()

    T_dot_13, issues_R_3 = simple_extract_1(T_dot_12)
    R_dot_13 = R_dot_12 + issues_R_3
    R_dot_13.sort()

    R_dot_14, issues_T_2 = simple_extract_3(R_dot_13)
    T_dot_14 = T_dot_13 + issues_T_2
    T_dot_14.sort()

    T_dot_15, issues_R_3 = simple_extract_1(T_dot_14)
    R_dot_15 = R_dot_14 + issues_R_3
    R_dot_15.sort()

    R_dot_16, issues_T_2 = simple_extract_3(R_dot_15)
    T_dot_16 = T_dot_15 + issues_T_2
    T_dot_16.sort()

    merged = sorted(R_dot_16 + T_dot_16)
    # 找出需要剔除的元素（后一个元素减前一个元素小于2）
    c = {merged[i + 1] for i in range(len(merged) - 1) if merged[i + 1] - merged[i] < 15}
    # 从a和b中剔除c中的元素
    R_dot_17 = [x for x in R_dot_16 if x not in c]
    T_dot_17 = [x for x in T_dot_16 if x not in c]

    T_dot_18, issues_R_3 = simple_extract_2(T_dot_17)
    R_dot_18 = R_dot_17 + issues_R_3
    R_dot_18.sort()

    R_dot_19, issues_T_2 = simple_extract_3(R_dot_18)
    T_dot_19 = T_dot_18 + issues_T_2
    T_dot_19.sort()

    T_dot_20, issues_R_3 = simple_extract_1(T_dot_19)
    R_dot_20 = R_dot_19 + issues_R_3
    R_dot_20.sort()

    R_dot_21, issues_T_2 = simple_extract_3(R_dot_20)
    T_dot_21 = T_dot_20 + issues_T_2
    T_dot_21.sort()

    T_dot_22, issues_R_3 = simple_extract_2(T_dot_21)
    R_dot_22 = R_dot_21 + issues_R_3
    R_dot_22.sort()

    R_dot_23, issues_T_2 = simple_extract_3(R_dot_22)
    T_dot_23 = T_dot_22 + issues_T_2
    T_dot_23.sort()

    T_dot_24, issues_R_3 = simple_extract_2(T_dot_23)
    R_dot_24 = R_dot_23 + issues_R_3
    R_dot_24.sort()
    R_dot_25, issues_T_2 = simple_extract_3(R_dot_24)
    T_dot_25 = T_dot_24 + issues_T_2
    T_dot_25.sort()


    R_dot_3 = R_dot_25
    T_dot_3 = T_dot_25

#///////////////////////////////
#检查手动R   T有没有错标
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


    def plot_ecg_segments(file_paths, ecg, R_dot_3, T_dot_3, ecg_size, start_index=0):
        """
        分段显示ECG文件数据
        参数:
        file_paths: 文件路径列表
        ecg: 合并后的ECG数据
        R_dot_3: R点位置列表
        T_dot_3: T点位置列表
        ecg_size: 每个文件的ECG数据长度
        start_index: 从第几个文件开始画 (默认从0开始)
        """
        # 确保起始索引在有效范围内
        start_index = max(0, min(start_index, len(file_paths) - 1))
        # 从指定索引开始遍历文件
        for i in range(start_index, len(file_paths)):
            plt.figure(figsize=(12, 6))
            # 计算当前文件在合并ECG中的起始和结束位置
            start_idx = i * ecg_size
            end_idx = start_idx + ecg_size
            # 提取当前文件的ECG数据
            current_ecg = ecg[start_idx:end_idx]
            # 提取当前文件的R和T点（在全局坐标中的位置）
            current_R = [r for r in R_dot_3 if start_idx <= r < end_idx]
            current_T = [t for t in T_dot_3 if start_idx <= t < end_idx]
            # 转换为局部坐标
            local_R = [r - start_idx for r in current_R]
            local_T = [t - start_idx for t in current_T]
            # 绘制当前文件的ECG
            plt.plot(current_ecg, 'k-', linewidth=0.8, alpha=0.7, label='ECG Signal')
            # 标记R点
            if local_R:
                plt.plot(local_R, [current_ecg[r] for r in local_R],
                         'ro', markersize=5, label='R波')
            # 标记T点
            if local_T:
                plt.plot(local_T, [current_ecg[t] for t in local_T],
                         'b^', markersize=5, label='T波')
            plt.xlabel('采样点')
            plt.ylabel('幅值')
            plt.title(f'ECG信号与检测到的R波和T波 - 文件 {i + 1} (总文件 {len(file_paths)})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            plt.close('all')
    # 使用示例：
    # 从第3个文件开始画（索引为2）
    plot_ecg_segments(file_paths, ecg, R_dot_3, T_dot_3, ecg_size, start_index=0)


#////////////////////////////////////

    # del dR[0]
    # del dT[0]
    # dR = process_and_combine_arrays(dR)
    # dT = process_and_combine_arrays(dT)
    # R = group_array(R_dot_3, fil_size)
    # T = group_array(T_dot_3, fil_size)
    #
    # # //////////////////////////////////
    # R_T = prediction.find_mutual_nearest_rt(ComputerR, ComputerT)
    # elements_to_remove = set()
    #
    # # 从 R 和 T 中删除在 elements_to_remove 中出现的元素
    # R_new = [x for x in ComputerR if x not in dR]
    # T_new = [x for x in ComputerT if x not in dT]
    # D_R_T_new = prediction.find_mutual_nearest_rt(R_new, T_new)
    # x_values = []
    # y_values = []
    # if len(R_T) >= len(D_R_T_new):
    #     # 找出在list1中但不在list2中的元组
    #     extra_tuples = [item for item in R_T if item not in D_R_T_new]
    #     result = [(x, y) for x, y in extra_tuples if abs(x - y) > 300]
    #     # 将多出来的元组的x和y分别放入不同的列表
    #     x_values = [item[0] for item in result]
    #     y_values = [item[1] for item in result]
    #
    # d_R_T_set = set(dR + dT)
    # # 1. 计算R数组内部元素之间的差值
    # R_pairs_with_small_diff = difference_value(ComputerR)
    # # 2. 计算T数组内部元素之间的差值
    # T_pairs_with_small_diff = difference_value(ComputerT)
    # # 3. 找出在d_R_T中的元素
    # R_in_d_R_T = multi(R_pairs_with_small_diff,d_R_T_set)
    # T_in_d_R_T = multi(T_pairs_with_small_diff,d_R_T_set)
    # R_in_d_R_T.extend(x_values)
    # T_in_d_R_T.extend(y_values)
    # R_in_d_R_T.sort()
    # T_in_d_R_T.sort()
    #
    # #///////////////////////////////
    #
    # R_T=prediction.find_mutual_nearest_rt(R_dot_3, T_dot_3)    #  手动RT匹配      T  波和  R波离得很近     否则find_mutual_nearest_rt()
    # missed_T_R = missed_detection(R_T, R_dot_3)      # 多的手R
    # missed_R_T = error_detection(R_T, T_dot_3)       #missed_detection      R_T(r,t)     漏的 手T
    # m_t_t = group_array(missed_R_T, fil_size)          #找出没有被匹配的点
    # m_r_r = group_array(missed_T_R, fil_size)
    # #R T对比
    # matching_R_CR, CR_matching, num_CR = prediction.get_rt_interval_RR(R_dot_3,ComputerR)    #坐标(R_dot,ComputerR)    差值
    # matching_R_DR, LR_matching, num_DR = prediction.get_rt_interval_RR(R_dot_3,lowerR)
    # matching_T_CT, CT_matching, num_CT = prediction.get_rt_interval_RR(T_dot_3,ComputerT)
    # matching_T_DT, LT_matching, num_DT = prediction.get_rt_interval_RR(T_dot_3, lowerT)
    # error_detection_Computer_R_point=error_detection(matching_R_CR,ComputerR)     #所有的误检
    # error_detection_Computer_T_point = error_detection(matching_T_CT, ComputerT)
    # error_detection_Downer_R_point = error_detection(matching_R_DR, lowerR)
    # error_detection_Downer_T_point=error_detection(matching_T_DT,lowerT)
    # err_C_R = group_array(error_detection_Computer_R_point,fil_size)      #单个帧的误检    6S数据改成6400
    # err_C_T = group_array(error_detection_Computer_T_point,fil_size)
    # err_D_R = group_array(error_detection_Downer_R_point,fil_size)
    # err_D_T = group_array(error_detection_Downer_T_point,fil_size)
    # dR = group_array(R_in_d_R_T,fil_size)
    # dT = group_array(T_in_d_R_T,fil_size)
    #
    # missed_Computer_R = missed_detection(matching_R_CR,R_dot_3)             #所有的漏检
    # missed_Computer_T = missed_detection(matching_T_CT, T_dot_3)
    # missed_Downer_R = missed_detection(matching_R_DR, R_dot_3)
    # missed_Downer_T = missed_detection(matching_T_DT, T_dot_3)
    # m_c_r = group_array(missed_Computer_R,fil_size)
    # m_c_t = group_array(missed_Computer_T,fil_size)
    # m_d_r = group_array(missed_Downer_R,fil_size)
    # m_d_t = group_array(missed_Downer_T,fil_size)
    #
    # a,total_percentages= group_and_count_d_arrays(CR_matching)    #用计算出来的手动标点和上位机标点的差值  计算出上位机R点和手动标点的 <50	50 - 100	100 - 150	150 - 200	>200百分比
    # print("------------上位机R点----------")
    # b,total_percentages1=group_and_count_d_arrays(LR_matching)
    # print("------------下位机R点----------")
    # c,total_percentages2=group_and_count_d_arrays(CT_matching)
    # print("------------上位机T点----------")
    # d,total_percentages3=group_and_count_d_arrays(LT_matching)
    # print("------------下位机T点----------")
    # data={
    #     "Manual_R_Num": len(R_dot_3),
    #     "Computer_R_Num": len(ComputerR),
    #     "Computer_R_matching_Num":num_CR,
    #     "Computer_R_deviation_percentage":total_percentages,
    #     "Computer_R_deviation_Num":a,
    #     "Downer_R_Num": len(lowerR),
    #     "Downer_R_matching_Num":num_DR,
    #     "Downer_R_deviation_percentage":  total_percentages1,
    #     "Downer_R_deviation_Num":b,
    #     "Manual_T_Num": len(T_dot_3) ,
    #     "Computer_T_Num": len(ComputerT),
    #     "Computer_T_matching_Num":num_CT,
    #     "Computer_T_deviation_percentage": total_percentages2,
    #     "Computer_T_deviation_Num":c,
    #     "Downer_T_Num": len(lowerT),
    #     "Downer_T_matching_Num":num_DT,
    #     "Downer_T_deviation_percentage": total_percentages3,
    #     "Downer_T_deviation_Num": d,
    #     "missed_Manual_T": (len(R_dot_3) - len(T_dot_3)),
    #     "missed_Computer_R": (len(R_dot_3) - num_CR),
    #     "missed_Computer_T": (len(R_dot_3) - num_CT),
    #     "missed_Downer_R": (len(R_dot_3) - num_DR),
    #     "missed_Downer_T": (len(R_dot_3) - num_DT),
    #     "error_detection_Computer_R": (len(ComputerR) - num_CR),
    #     "error_detection_Computer_T": (len(ComputerT) - num_CT),
    #     "error_detection_Downer_R": (len(lowerR) - num_DR),
    #     "error_detection_Downer_T": (len(lowerT) - num_DT),
    #     "deleted_R_point":  sum(len(sublist) for sublist in dR),
    #     "deleted_T_point": sum(len(sublist) for sublist in dT)
    # }
    # file_path9 = base_path+ "Overall.txt"
    # with open(file_path9, 'w', encoding='utf-8') as file:
    #     json.dump(data, file, indent=4, ensure_ascii=False)
    # del CRTIL[0:2]
    # del RTIL[0]
    # data_Interval={
    #     "Manual_Interval":MRTIL,    #手动标点的间隔
    #     "Practical_Interval":CRTIL,   #实际存下来的数据的间隔
    #     "issue_Interval":RTIL         #上位机下发的间隔
    # }
    # file_path10 = base_path+"RTInterval.txt"
    # with open(file_path10, 'w', encoding='utf-8') as file:
    #     json.dump(data_Interval, file, ensure_ascii=False)
    #
    # ComputerR = grouping(matching_R_CR)
    # lowerR = grouping(matching_R_DR)
    # ComputerT = grouping(matching_T_CT)
    # lowerT = grouping(matching_T_DT)
    # dR.append([])
    # dT.append([])
    # i=0
    # for path in file_paths:
    #     file_path_root = path
    #     Num_C_R, C_R_array = group_and_count_2d_arrays(ComputerR[i])
    #     Num_C_T, C_T_array = group_and_count_2d_arrays(ComputerT[i])
    #     Num_D_R, D_R_array = group_and_count_2d_arrays(lowerR[i])
    #     Num_D_T, D_T_array = group_and_count_2d_arrays(lowerT[i])
    #
    #     e_C_R_N, e_C_R_P = count_elements(err_C_R,i)
    #     e_D_R_N, e_D_R_P = count_elements(err_D_R,i)
    #     e_C_T_N, e_C_T_P = count_elements(err_C_T,i)
    #     e_D_T_N, e_D_T_P = count_elements(err_D_T,i)
    #
    #     m_C_R_N, m_C_R_P = count_elements(m_c_r, i)
    #     m_D_R_N, m_D_R_P = count_elements(m_d_r, i)
    #     m_C_T_N, m_C_T_P = count_elements(m_c_t, i)
    #     m_D_T_N, m_D_T_P = count_elements(m_d_t, i)
    #     m_R_T_N, m_R_T_P = count_elements(m_t_t, i)
    #     file_path11 = file_path_root + "/interior_Interval.txt"
    #     if os.path.exists(file_path11):
    #         with open(file_path11, 'r', encoding='utf-8') as file:
    #             try:
    #                 existing_data = json.load(file)
    #             except json.JSONDecodeError:
    #                 existing_data = {}
    #     else:
    #         existing_data = {}
    #     date2 = {
    #         "Manual_R_Num": len(R[i]),  # 手动R
    #         "Manual_T_Num": len(T[i]),  # 手动T
    #         "missed_Manual_T": m_R_T_N,      #应该是误检
    #         "Computer_R_matching_Num": len(ComputerR[i]),
    #         "Downer_R_matching_Num":len(lowerR[i]),
    #         "Computer_R_deviation_Num":Num_C_R,
    #         "Downer_R_deviation_Num":Num_D_R,
    #         "Computer_T_matching_Num": len(ComputerT[i]),
    #         "Downer_T_matching_Num": len(lowerT[i]),
    #         "Computer_T_deviation_Num":Num_C_T,
    #         "Downer_T_deviation_Num":Num_D_T,
    #         "missed_Computer_R": m_C_R_N,
    #         "missed_Downer_R": m_D_R_N,
    #         "missed_Computer_T": m_C_T_N,
    #         "missed_Downer_T": m_D_T_N,
    #         "error_detection_Computer_R": e_C_R_N,
    #         "error_detection_Computer_T": e_C_T_N,
    #         "error_detection_Downer_R": e_D_R_N,
    #         "error_detection_Downer_T": e_D_T_N,
    #         "deleted_R_point": len(dR[i]),
    #         "deleted_T_point": len(dT[i])
    #     }
    #     # ///////////
    #     existing_data.update(date2)
    #     with open(file_path11, 'w', encoding='utf-8') as file:
    #         json.dump(existing_data, file, indent=4, ensure_ascii=False)
    #     file_path = file_path_root + "/gauge_point.txt"
    #     date = {
    #         "missed_Manual_T_point":m_R_T_P,
    #         "Computer_R": C_R_array,
    #         "Computer_T": C_T_array,
    #         "Downer_R":D_R_array,
    #         "Downer_T":D_T_array,
    #         "error_detection_Computer_R_point":e_C_R_P,
    #         "error_detection_Computer_T_point":e_C_T_P,
    #         "error_detection_Downer_R_point":e_D_R_P,
    #         "error_detection_Downer_T_point":e_D_T_P,
    #         "missed_Computer_R_point": m_C_R_P,
    #         "missed_Downer_R_point": m_D_R_P,
    #         "missed_Computer_T_point": m_C_T_P,
    #         "missed_Downer_T_point": m_D_T_P,
    #         "deleted_Computer_R_point":dR[ i ],
    #         "deleted_Computer_T_point": dT[i ]
    #             }
    #
    #     with open(file_path, 'w', encoding='utf-8') as file:
    #         json.dump(date, file,  ensure_ascii=False)
    #     file_path8 = file_path_root + "/R_DetectedByManual.txt"
    #     data_R = {"R_DetectedByManual": R[i]}
    #     with open(file_path8, 'w', encoding='utf-8') as file:
    #         json.dump(data_R, file, indent=4, ensure_ascii=False)
    #
    #     file_path15 = file_path_root + "/T_DetectedByManual.txt"
    #     data_T = {"T_DetectedByManual": T[i]}
    #     with open(file_path15, 'w', encoding='utf-8') as file:
    #         json.dump(data_T, file, indent=4, ensure_ascii=False)
    #     i = i + 1


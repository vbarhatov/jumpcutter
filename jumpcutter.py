import subprocess
from audiotsm.io.wav import WavReader, WavWriter
from scipy.io import wavfile
import numpy as np
import re
import math
from shutil import rmtree, move
import os
import argparse
from pytube import YouTube
from time import time

def downloadFile(url):
    sep = os.path.sep
    originalPath = YouTube(url).streams.first().download()
    filepath = originalPath.split(sep)
    filepath[-1] = filepath[-1].replace(' ','_')
    filepath = sep.join(filepath)
    os.rename(originalPath, filepath)
    return filepath

def getFrameRate(path):
    process = subprocess.Popen(["ffmpeg", "-i", path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, _ = process.communicate()
    output =  stdout.decode()
    match_dict = re.search(r"\s(?P<fps>[\d\.]+?)\stbr", output).groupdict()
    return float(match_dict["fps"])

def getMaxVolume(s):
    maxv = float(np.max(s))
    minv = float(np.min(s))
    return max(maxv,-minv)

def copyFrame(inputFrame, outputFrame, tempDir):
    src = tempDir+"/frame{:06d}".format(inputFrame+1)+".jpg"
    dst = tempDir+"/newFrame{:06d}".format(outputFrame+1)+".jpg"
    if not os.path.isfile(src):
        return False
    move(src, dst)
    if outputFrame % 1000 == 999:
        print(str(outputFrame + 1) + " time-altered frames saved.")
    return True

def inputToOutputFilename(filename):
    dotIndex = filename.rfind(".")
    return filename[:dotIndex]+"_ALTERED"+filename[dotIndex:]

def createTempDir(parentDir):
    tempDir = parentDir + "/temp" + str(int(time()))
    try:  
        os.mkdir(tempDir)
        return tempDir
    except OSError:  
        assert False, "Creation of the directory %s failed. (The TEMP folder may already exist. Delete or rename it, and try again.)"

def deletePath(s): # Dangerous! Watch out!
    try:  
        rmtree(s,ignore_errors=False)
    except OSError:  
        print ("Deletion of the directory %s failed" % s)
        print(OSError)

def getVideoLengthSeconds(filename):
    output = subprocess.check_output(("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filename)).strip()
    return int(float(output))

def copyVideoFragment(inputFile, outputFile, startSeconds, endSeconds):
    # ffmpeg -i input.mp4 -vcodec copy -acodec copy -ss 0 -t 30 output.mp4
    split_cmd = ["ffmpeg", "-i", inputFile, "-vcodec", "copy", "-acodec", "copy"]
    if startSeconds > 0:
        split_cmd.extend(["-ss", str(startSeconds)])
    split_cmd.extend(["-t", str(endSeconds - startSeconds), outputFile])
    subprocess.check_output(split_cmd)

def processVideo(inputFile, outputFile, tempDir):
    global frameRate
    command = "ffmpeg -i '" + inputFile + "' -qscale:v " + str(FRAME_QUALITY) + " " + tempDir + "/frame%06d.jpg -hide_banner"
    subprocess.call(command, shell=True)
    command = "ffmpeg -i '" + inputFile + "' -ab 160k -ac 2 -ar " + str(SAMPLE_RATE) + " -vn " + tempDir + "/audio.wav"
    subprocess.call(command, shell=True)
    sampleRate, audioData = wavfile.read(tempDir + "/audio.wav")
    audioSampleCount = audioData.shape[0]
    maxAudioVolume = getMaxVolume(audioData)
    if frameRate is None:
        frameRate = getFrameRate(inputFile)
    samplesPerFrame = sampleRate / frameRate
    audioFrameCount = int(math.ceil(audioSampleCount / samplesPerFrame))
    hasLoudAudio = np.zeros((audioFrameCount))
    for i in range(audioFrameCount):
        start = int(i * samplesPerFrame)
        end = min(int((i + 1) * samplesPerFrame), audioSampleCount)
        audiochunks = audioData[start:end]
        maxchunksVolume = float(getMaxVolume(audiochunks)) / max(maxAudioVolume, 1e-10)
        if maxchunksVolume >= SILENT_THRESHOLD:
            hasLoudAudio[i] = 1
    chunks = [[0, 0, 0]]
    shouldIncludeFrame = np.zeros((audioFrameCount))
    for i in range(audioFrameCount):
        start = int(max(0, i - FRAME_SPREADAGE))
        end = int(min(audioFrameCount, i + 1 + FRAME_SPREADAGE))
        shouldIncludeFrame[i] = np.max(hasLoudAudio[start:end])
        if (i >= 1 and shouldIncludeFrame[i] != shouldIncludeFrame[i - 1]):  # Did we flip?
            chunks.append([chunks[-1][1], i, shouldIncludeFrame[i - 1]])
    chunks.append([chunks[-1][1], audioFrameCount, shouldIncludeFrame[i - 1]])
    chunks = chunks[1:]
    outputAudioData = []
    outputPointer = 0
    mask = [x / AUDIO_FADE_ENVELOPE_SIZE for x in range(AUDIO_FADE_ENVELOPE_SIZE)]  # Create audio envelope mask
    lastExistingFrame = None
    for chunk in chunks:
        audioChunk = audioData[int(chunk[0] * samplesPerFrame):int(chunk[1] * samplesPerFrame)]

        sFile = tempDir + "/tempStart.wav"
        eFile = tempDir + "/tempEnd.wav"
        wavfile.write(sFile, SAMPLE_RATE, audioChunk)
        with WavReader(sFile) as reader:
            with WavWriter(eFile, reader.channels, reader.samplerate) as writer:
                tsm = audio_stretch_algorithm(reader.channels, speed=NEW_SPEED[int(chunk[2])])
                tsm.run(reader, writer)
        _, alteredAudioData = wavfile.read(eFile)
        leng = alteredAudioData.shape[0]
        endPointer = outputPointer + leng
        outputAudioData.extend((alteredAudioData / maxAudioVolume).tolist())

        # Smoothing the audio
        if leng < AUDIO_FADE_ENVELOPE_SIZE:
            for i in range(outputPointer, endPointer):
                outputAudioData[i] = 0
        else:
            for i in range(outputPointer, outputPointer + AUDIO_FADE_ENVELOPE_SIZE):
                outputAudioData[i][0] *= mask[i - outputPointer]
                outputAudioData[i][1] *= mask[i - outputPointer]
            for i in range(endPointer - AUDIO_FADE_ENVELOPE_SIZE, endPointer):
                outputAudioData[i][0] *= (1 - mask[i - endPointer + AUDIO_FADE_ENVELOPE_SIZE])
                outputAudioData[i][1] *= (1 - mask[i - endPointer + AUDIO_FADE_ENVELOPE_SIZE])

        startOutputFrame = int(math.ceil(outputPointer / samplesPerFrame))
        endOutputFrame = int(math.ceil(endPointer / samplesPerFrame))
        for outputFrame in range(startOutputFrame, endOutputFrame):
            inputFrame = int(chunk[0] + NEW_SPEED[int(chunk[2])] * (outputFrame - startOutputFrame))
            didItWork = copyFrame(inputFrame, outputFrame, tempDir)
            if didItWork:
                lastExistingFrame = inputFrame
            else:
                copyFrame(lastExistingFrame, outputFrame, tempDir)

        outputPointer = endPointer
    outputAudioData = np.asarray(outputAudioData)
    wavfile.write(tempDir + "/audioNew.wav", SAMPLE_RATE, outputAudioData)
    command = f"ffmpeg -framerate {frameRate} -i {tempDir}/newFrame%06d.jpg -i {tempDir}/audioNew.wav -strict -2 -c:v libx264 -preset {H264_PRESET} -crf {H264_CRF} -pix_fmt yuvj420p '{outputFile}'"
    subprocess.call(command, shell=True)


def joinVideos(inputFiles, outputFile, tempDir):
    # $ cat mylist.txt
    # file '/path/to/file1'
    # file '/path/to/file2'
    # file '/path/to/file3'
    #
    # $ ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.mp4
    outputListPath = '{}/outputList.txt'.format(tempDir)
    outputListFile = open(outputListPath, 'w+')
    for file in inputFiles:
        outputListFile.write('file \'{}\'\n'.format(file))
    outputListFile.close()
    joinCmd = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", outputListPath, "-c", "copy", outputFile]
    subprocess.check_output(joinCmd)
    pass

def joinVideos1(inputFiles, outputFile):
    # e.g.: ffmpeg -i "concat:input1.mp4|input2.mp4|input3.mp4" -c copy output.mp4
    joinedInputFiles = '|'.join(inputFiles)
    split_cmd = ["ffmpeg", "-i", "concat:{}".format(joinedInputFiles), "-c", "copy", outputFile]
    subprocess.check_output(split_cmd)
    pass



parser = argparse.ArgumentParser(description='Modifies a video file to play at different speeds when there is sound vs. silence.')
parser.add_argument('-i', '--input_file', type=str,  help='the video file you want modified')
parser.add_argument('-u', '--url', type=str, help='A youtube url to download and process')
parser.add_argument('-o', '--output_file', type=str, default="", help="the output file. (optional. if not included, it'll just modify the input file name)")
parser.add_argument('-slt', '--silent_threshold', type=float, default=0.03, help="the volume amount that frames' audio needs to surpass to be consider \"sounded\". It ranges from 0 (silence) to 1 (max volume)")
parser.add_argument('-sns', '--sounded_speed', type=float, default=1.70, help="the speed that sounded (spoken) frames should be played at. Typically 1.")
parser.add_argument('-sls', '--silent_speed', type=float, default=8.00, help="the speed that silent frames should be played at. 999999 for jumpcutting.")
parser.add_argument('-fm', '--frame_margin', type=float, default=1, help="some silent frames adjacent to sounded frames are included to provide context. How many frames on either the side of speech should be included? That's this variable.")
parser.add_argument('-sr', '--sample_rate', type=int, default=44100, help="sample rate of the input and output videos")
parser.add_argument('-fr', '--frame_rate', type=float, help="frame rate of the input and output videos. optional... I try to find it out myself, but it doesn't always work.")
parser.add_argument('-fq', '--frame_quality', type=int, default=3, help="quality of frames to be extracted from input video. 1 is highest, 31 is lowest, 3 is the default.")
parser.add_argument('-p', '--preset', type=str, default="medium", help="A preset is a collection of options that will provide a certain encoding speed to compression ratio. See https://trac.ffmpeg.org/wiki/Encode/H.264")
parser.add_argument('-crf', '--crf', type=int, default=23, help="Constant Rate Factor (CRF). Lower value - better quality but large filesize. See https://trac.ffmpeg.org/wiki/Encode/H.264")
parser.add_argument('-sa', '--stretch_algorithm', type=str, default="wsola", help="Sound stretching algorithm. 'phasevocoder' is best in general, but sounds phasy. 'wsola' may have a bit of wobble, but sounds better in many cases.")


args = parser.parse_args()


frameRate = args.frame_rate
SAMPLE_RATE = args.sample_rate
SILENT_THRESHOLD = args.silent_threshold
FRAME_SPREADAGE = args.frame_margin
NEW_SPEED = [args.silent_speed, args.sounded_speed]
if args.url != None:
    INPUT_FILE = downloadFile(args.url)
else:
    INPUT_FILE = args.input_file
URL = args.url
FRAME_QUALITY = args.frame_quality
H264_PRESET = args.preset
H264_CRF = args.crf

STRETCH_ALGORITHM = args.stretch_algorithm
if(STRETCH_ALGORITHM == "phasevocoder"):
    from audiotsm import phasevocoder as audio_stretch_algorithm
elif (STRETCH_ALGORITHM == "wsola"):
    from audiotsm import wsola as audio_stretch_algorithm
else:
    raise Exception("Unknown audio stretching algorithm.")

assert INPUT_FILE != None , "why u put no input file, that dum"
assert FRAME_QUALITY < 32 , "The max value for frame quality is 31."
assert FRAME_QUALITY > 0 , "The min value for frame quality is 1."
    
if len(args.output_file) >= 1:
    OUTPUT_FILE = args.output_file
else:
    OUTPUT_FILE = inputToOutputFilename(INPUT_FILE)

AUDIO_FADE_ENVELOPE_SIZE = 400 # smooth out transitiion's audio by quickly fading in/out (arbitrary magic number whatever)


tempDir = createTempDir("/tmp")

partSizeSeconds = 600
videoLengthSeconds = getVideoLengthSeconds(INPUT_FILE)

if videoLengthSeconds < partSizeSeconds:
    processVideo(INPUT_FILE, OUTPUT_FILE, tempDir)
else:
    videoPartStart = 0
    i = 0
    outputFileParts = []
    while videoPartStart < videoLengthSeconds:
        print('\n\nPROCESSING PART {}/{}\n\n'.format(i + 1, math.ceil(videoLengthSeconds / partSizeSeconds)))
        partTempDir = createTempDir(tempDir)
        videoPartEnd = min(videoLengthSeconds, videoPartStart + partSizeSeconds)
        inputFilePart = "{}/inputPart.{:03d}.mp4".format(tempDir, i)
        outputFilePart = "{}/outputPart.{:03d}.mp4".format(tempDir, i)
        outputFileParts.append(outputFilePart)
        copyVideoFragment(INPUT_FILE, inputFilePart, videoPartStart, videoPartEnd)
        processVideo(inputFilePart, outputFilePart, partTempDir)

        videoPartStart = videoPartEnd
        i += 1

        deletePath(partTempDir)
    joinVideos(outputFileParts, OUTPUT_FILE, tempDir)

deletePath(tempDir)


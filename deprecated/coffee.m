% Main script for the experimental setup
% Author: Your Name
% Date: Today
% This script runs Stroop, Go/No-Go, N-back, BART, and Emotion tasks,
% balanced in order using a Latin square. Each participant does tasks
% before and after coffee intake.

% Import Psychtoolbox
Screen('Preference', 'SkipSyncTests', 1); % Skip sync tests for debugging

% Input participant ID
participantID = input('Enter participant ID: ', 's');
session = input('Enter session (1 = Pre-coffee, 2 = Post-coffee): ');

% Initialize randomization seed
rng('shuffle');

% Define tasks and Latin square order
tasks = {'Stroop', 'GoNoGo', 'NBack', 'BART', 'Emotion'};
numTasks = numel(tasks);

% Generate Latin square order
latinSquare = zeros(numTasks, numTasks);
for i = 1:numTasks
    latinSquare(i, :) = mod((0:numTasks-1) + (i-1), numTasks) + 1;
end

% Assign a specific order based on participant ID
participantOrderIndex = mod(str2double(participantID), numTasks) + 1;
taskOrder = latinSquare(participantOrderIndex, :);

% Display task order for debugging
disp(['Task order for participant ', participantID, ': ', strjoin(tasks(taskOrder), ', ')]);

% Run tasks in the determined order
for taskIdx = 1:numTasks
    taskName = tasks{taskOrder(taskIdx)};
    disp(['Starting task: ', taskName]);
    switch taskName
        case 'Stroop'
            runStroop(participantID, session);
        case 'GoNoGo'
            runGoNoGo(participantID, session);
        case 'NBack'
            runNBack(participantID, session);
        case 'BART'
            runBART(participantID, session);
        case 'Emotion'
            runEmotion(participantID, session);
    end
end

disp('All tasks completed!');

%% Task Functions

%% Stroop Task
function runStroop(participantID, session)
    % Stroop任务：评估干扰抑制能力
    % Inputs:
    %   participantID - 被试编号
    %   session - 实验阶段（1: 咖啡摄入前, 2: 咖啡摄入后）

    % 设置实验参数
    numTrials = 8;           % 每个trial的轮数
    numStimuliPerTrial = 12; % 每个trial中呈现的刺激次数
    stimDuration = 0.5;      % 每个刺激的呈现时间 (秒)
    isi = 1;                 % 刺激间隔时间 (秒)
    numPracticeStimuli = 6;  % 练习阶段的刺激次数

    % 定义颜色和文字条件
    colors = {'Red', 'Green', 'Blue'};   % 刺激的颜色
    words = {'红色', '绿色', '蓝色'};    % 刺激的文字
    neutralWord = '白色';                % 中性文字

    % 定义9种条件
    conditions = [
        % word  color
        1       1;  % 同色
        2       2;
        3       3;
        1       2;  % 异色
        2       3;
        3       1;
        0       1;  % 中性词
        0       2;
        0       3
    ];

    % 初始化 Psychtoolbox
    Screen('Preference', 'SkipSyncTests', 1);  % 跳过同步测试（调试时使用）
    [win, rect] = Screen('OpenWindow', max(Screen('Screens')), [128 128 128]);  % 灰背景
    [screenWidth, screenHeight] = Screen('WindowSize', win); % 获取屏幕尺寸
    Screen('TextFont', win, 'Arial');
    Screen('TextSize', win, round(screenHeight / 5));  % 字体大小设为屏幕高度的五分之一
    [xCenter, yCenter] = RectCenter(rect);


% 保存原来的字体大小设置
originalTextSize = Screen('TextSize', win);

% 设置指导语的字体大小为24
Screen('TextSize', win, 24);

% 显示指导语
instructions = ['在这个任务中，你需要不在意这些词语的字义，只在意这些文字的颜色。\n\n' ...
                '当词语的颜色是红色的时候，按1；绿色的时候，按2，蓝色的时候，按3。\n' ...
                '按任意键开始实验'];

% 使用 Screen('DrawText') 替代 DrawFormattedText
% 将长文本分成多行显示
lines = strsplit(instructions, '\n');
lineHeight = 30;  % 行间距
startY = yCenter - (length(lines) * lineHeight) / 2;  % 从中心开始向上偏移

for i = 1:length(lines)
    thisLine = lines{i};
    % 获取文本宽度以居中显示
    [normBoundsRect, ~] = Screen('TextBounds', win, thisLine);
    textWidth = normBoundsRect(3) - normBoundsRect(1);
    textX = xCenter - textWidth/2;
    textY = startY + (i-1) * lineHeight;
    Screen('DrawText', win, thisLine, textX, textY, [255 255 255]);
end

Screen('Flip', win);
KbStrokeWait;

% 恢复来的字体大小
Screen('TextSize', win, originalTextSize);

    %% 练习阶段
    disp('Starting practice trials...');
    for stim = 1:numPracticeStimuli
        % 随机选择一个条件
        condition = conditions(randi(size(conditions, 1)), :);
        wordIdx = condition(1);
        colorIdx = condition(2);

        if wordIdx == 0  % 中性词
            word = neutralWord;
        else
            word = words{wordIdx};
        end
        color = colors{colorIdx};

        % 显示刺激
        DrawFormattedText(win, word, 'center', 'center', getColor(color));
        Screen('Flip', win);
        WaitSecs(stimDuration);  % 显示刺激的时间

        % 清空屏幕，等待按键
        Screen('Flip', win);
        waitForResponse();  % 等待被试按键

        % 间隔时间
        WaitSecs(isi);
    end
    disp('Practice trials completed.');

    % 提示开始正式实验
    instructions = 'Now the actual experiment begins. Press any key to continue.';
    DrawFormattedText(win, instructions, 'center', 'center', [255 255 255]);
    Screen('Flip', win);
    KbStrokeWait;

    %% 正式实验阶段
    % 初始化记录变量
    responseTimes = zeros(numTrials, numStimuliPerTrial);
    correctResponses = zeros(numTrials, numStimuliPerTrial);
    conditionLabels = zeros(numTrials, numStimuliPerTrial);  % 标记每个刺激属于哪种条件

    % 开始实验
    for trial = 1:numTrials
        % 当前 trial 的随机条件顺序
        trialConditions = conditions(randi(size(conditions, 1), numStimuliPerTrial, 1), :);

        for stim = 1:numStimuliPerTrial
            % 获取当前刺激的文字和颜色
            wordIdx = trialConditions(stim, 1);
            colorIdx = trialConditions(stim, 2);

            if wordIdx == 0  % 中性词
                word = neutralWord;
            else
                word = words{wordIdx};
            end
            color = colors{colorIdx};

            % 显示刺激
            DrawFormattedText(win, word, 'center', 'center', getColor(color));
            Screen('Flip', win);
            stimulusOnset = GetSecs;

            % 等待刺激持续时间
            WaitSecs(stimDuration);

            % 清空屏幕
            Screen('Flip', win);

            % 等待按键响应
            [responseTime, isCorrect] = waitForResponseWithTiming(stimulusOnset, color);
            responseTimes(trial, stim) = responseTime;
            correctResponses(trial, stim) = isCorrect;

            % 标记条件（1=同色, 2=异色, 3=中性）
            if wordIdx == colorIdx && wordIdx ~= 0
                conditionLabels(trial, stim) = 1;  % 同色
            elseif wordIdx ~= colorIdx && wordIdx ~= 0
                conditionLabels(trial, stim) = 2;  % 异色
            else
                conditionLabels(trial, stim) = 3;  % 中性
            end

            % 间隔时间
            WaitSecs(isi);
        end
    end

    % 关闭 Psychtoolbox
    Screen('CloseAll');

    % 数据统计：按条件分类计算
    data.responseTimes = responseTimes;
    data.correctResponses = correctResponses;
    data.conditionLabels = conditionLabels;

    % 分类统计反应时和正确率
    data.meanRT_same = nanmean(responseTimes(conditionLabels == 1));
    data.meanRT_diff = nanmean(responseTimes(conditionLabels == 2));
    data.meanRT_neutral = nanmean(responseTimes(conditionLabels == 3));
    data.acc_same = nanmean(correctResponses(conditionLabels == 1)) * 100;
    data.acc_diff = nanmean(correctResponses(conditionLabels == 2)) * 100;
    data.acc_neutral = nanmean(correctResponses(conditionLabels == 3)) * 100;

    % 保存数据
    save(['Stroop_' participantID '_session' num2str(session) '.mat'], 'data');

    disp(['Stroop task completed for participant ', participantID, ' in session ', num2str(session)]);
end

%% Go/No-Go Task
function runGoNoGo(participantID, session)
    % Implement Go/No-Go task here
    disp(['Running Go/No-Go task for participant ', participantID, ' in session ', num2str(session)]);
    % Your Go/No-Go task logic
end

%% N-Back Task
function runNBack(participantID, session)
    % N-back任务：评估工作记忆
    % Inputs:
    %   participantID - 被试编号
    %   session - 实验阶段（1: 咖啡摄入前, 2: 咖啡摄入后）
    
    % 实验参数设置
    stimDuration = 1;  % 刺激呈现时间（秒）
    isi = 1.5;          % 刺激间隔时间（秒）
    trialsPerBlock = 30;  % 每个block的试次数
    numBlocksPerCondition = 5;  % 每个条件的block数
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';  % 刺激字母池
    
    % 添加练习参数
    practiceTrialsPerBlock = 15;  % 练习时每个block的试次数
    numPracticeBlocksPerCondition = 1;  % 每个条件的练习block数
    
    % 初始化Psychtoolbox
    Screen('Preference', 'SkipSyncTests', 1);
    [win, rect] = Screen('OpenWindow', max(Screen('Screens')), [128 128 128]);  % 灰背景
    [screenWidth, screenHeight] = Screen('WindowSize', win);
    Screen('TextFont', win, 'Arial');
    Screen('TextSize', win, round(screenHeight / 10));
    [xCenter, yCenter] = RectCenter(rect);
    
    % 显示实验说明
    ShowInstructions(win);
    
    % 运行练习block
    practiceBlockTypes = [0, 1, 2];  % 0-back, 1-back, 2-back的练习
    for blockNum = 1:length(practiceBlockTypes)
        nback = practiceBlockTypes(blockNum);
        
        % 显示练习block开始提示
        instructions = sprintf('练习阶段：%d-back任务\n\n', nback);
        switch nback
            case 0
                instructions = [instructions '当看到目标字母时按J键\n按任意键开始练习'];
            case 1
                instructions = [instructions '当当前字母与前一个字母相同时按J键\n按任意键开始练习'];
            case 2
                instructions = [instructions '当当前字母与前两个字母相同时按J键\n按任意键开始练习'];
        end
        
        Screen('TextSize', win, 24);
        DrawFormattedText(win, instructions, 'center', 'center', [255 255 255]);
        Screen('Flip', win);
        KbStrokeWait;
        Screen('TextSize', win, round(screenHeight / 10));
        
        % 生成练习刺激序列
        stimuli = GenerateStimuliSequence(letters, practiceTrialsPerBlock, nback);
        targetPositions = zeros(1, practiceTrialsPerBlock);
        
        if nback == 0
            targetLetter = stimuli(1);
            targetPositions = strcmp(stimuli, targetLetter);
        else
            for i = (nback+1):practiceTrialsPerBlock
                if strcmp(stimuli{i}, stimuli{i-nback})
                    targetPositions(i) = 1;
                end
            end
        end
        
        % 运行练习trials（不记录数据）
        RunPracticeTrials(win, stimuli, targetPositions, stimDuration, isi, nback);
        
        % 显示练习结束提示
        DrawFormattedText(win, '练习结束\n\n按任意键继续', 'center', 'center', [255 255 255]);
        Screen('Flip', win);
        KbStrokeWait;
    end
    
    % 显示正式实验开始提示
    DrawFormattedText(win, '练习阶段已结束\n\n现在开始正式实验\n\n按任意键继续', 'center', 'center', [255 255 255]);
    Screen('Flip', win);
    KbStrokeWait;
    
    % 初始化数据记录
    data = struct();
    blockTypes = [zeros(1, numBlocksPerCondition), ...    % 0-back blocks
                  ones(1, numBlocksPerCondition), ...     % 1-back blocks
                  zeros(1, numBlocksPerCondition), ...    % 0-back blocks again
                  2 * ones(1, numBlocksPerCondition)];    % 2-back blocks
    
    % 运行所有block
    for blockNum = 1:length(blockTypes)
        nback = blockTypes(blockNum);
        
        % 显示block开始提示
        switch nback
            case 0
                instructions = '0-back任务：\n当看到目标字母时按J键\n按任意键开始';
            case 1
                instructions = '1-back任务：\n当当前字母与前一个字母相同时按J键\n按任意键开始';
            case 2
                instructions = '2-back任务：\n当当前字母与前两个字母相同时按J键\n按任意键开始';
        end
        
        Screen('TextSize', win, 24);
        DrawFormattedText(win, instructions, 'center', 'center', [255 255 255]);
        Screen('Flip', win);
        KbStrokeWait;
        Screen('TextSize', win, round(screenHeight / 10));
        
        % 生成刺激序列
        stimuli = GenerateStimuliSequence(letters, trialsPerBlock, nback);
        targetPositions = zeros(1, trialsPerBlock);
        
        if nback == 0
            targetLetter = stimuli(1);  % 使用第一个字母作为目标
            targetPositions = strcmp(stimuli, targetLetter);
        else
            for i = (nback+1):trialsPerBlock
                if strcmp(stimuli{i}, stimuli{i-nback})
                    targetPositions(i) = 1;
                end
            end
        end
        
        % 运行当前block的试次
        [rt, accuracy] = RunTrials(win, stimuli, targetPositions, stimDuration, isi, nback);
        
        % 保存数据
        data(blockNum).nback = nback;
        data(blockNum).rt = rt;
        data(blockNum).accuracy = accuracy;
        data(blockNum).hits = sum(rt > 0 & targetPositions);
        data(blockNum).falseAlarms = sum(rt > 0 & ~targetPositions);
        data(blockNum).misses = sum(~rt > 0 & targetPositions);
        data(blockNum).correctRejections = sum(~rt > 0 & ~targetPositions);
        data(blockNum).meanRT = mean(rt(rt > 0));
        data(blockNum).stdRT = std(rt(rt > 0));
    end
    
    % 关闭窗口
    Screen('CloseAll');
    
    % 保存数据
    save(['NBack_' participantID '_session' num2str(session) '.mat'], 'data');
    
    disp(['N-back task completed for participant ', participantID, ' in session ', num2str(session)]);
end

function ShowInstructions(win)
    instructions = ['N-back任务说明：\n\n' ...
                   '在这个任务中，你将看到一系列字母。\n' ...
                   '任务分为三种类型：\n\n' ...
                   '0-back：当看到指定的目标字母时按J键\n' ...
                   '1-back：当当前字母与前一个字母相同时按J键\n' ...
                   '2-back：当当前字母与前两个字母相同时按J键\n\n' ...
                   '按任意键继续'];
    
    Screen('TextSize', win, 24);
    DrawFormattedText(win, instructions, 'center', 'center', [255 255 255]);
    Screen('Flip', win);
    KbStrokeWait;
    Screen('TextSize', win, round(screenHeight / 10));
end

%% BART Task
function runBART(participantID, session)
    % BART任务：评估风险决策行为
    % Inputs:
    %   participantID - 被试编号
    %   session - 实验阶段（1: 咖啡摄入前, 2: 咖啡摄入后）
    
    % 实验参数设置
    numTrials = 3;           % 气球总数为3
    initialSize = 50;        % 初始气球大小（像素）
    maxSize = 300;          % 最大气球大小
    growthRate = 10;        % 每次充气增加的大小
    moneyPerPump = 5;       % 每次充气奖励金额为5元
    baseExplosionProb = 0.05;  % 基础爆炸概率
    probIncrease = 0.1;    % 每次充气增加的爆炸概率
    
    % 初始化Psychtoolbox
    Screen('Preference', 'SkipSyncTests', 1);
    [win, rect] = Screen('OpenWindow', max(Screen('Screens')), [0 0 0]);  % 黑背景
    [screenWidth, screenHeight] = Screen('WindowSize', win);
    [xCenter, yCenter] = RectCenter(rect);
    
    % 显示实验说明
    Screen('TextSize', win, 24);
    instructions = ['气球风险任务(BART)\n\n' ...
                   '在这个任务中，你将看到一个气球。\n' ...
                   '按空格键给气球充气，每次充气可以赚取3元。\n' ...
                   '但是气球可能会在任何时候爆炸，爆炸后将失去本轮所有奖励。\n' ...
                   '按Enter键可以随时结束当前轮次，保住已经赚到的钱。\n\n' ...
                   '按任意键开始实验'];
    DrawFormattedText(win, instructions, 'center', 'center', [255 255 255]);
    Screen('Flip', win);
    KbStrokeWait;
    
    % 初始化数据记录
    data = struct();
    data.numPumps = zeros(1, numTrials);
    data.earnings = zeros(1, numTrials);
    data.explosions = zeros(1, numTrials);
    
    % 运行trials
    for trial = 1:numTrials
        % 初始化当前trial的变量
        currentSize = initialSize;
        currentMoney = 0;
        numPumps = 0;
        exploded = false;
        currentProb = baseExplosionProb;
        
        % 显示trial数
        trialText = sprintf('气球 %d/%d', trial, numTrials);
        Screen('TextSize', win, 20);
        DrawFormattedText(win, trialText, 'center', screenHeight * 0.1, [255 255 255]);
        
        % trial循环
        while ~exploded
            % 绘制气球
            balloonColor = [255 0 0];  % 红色气球
            balloonRect = CenterRectOnPoint([0 0 currentSize currentSize], xCenter, yCenter);
            Screen('FillOval', win, balloonColor, balloonRect);
            
            % 显示当前金额
            moneyText = sprintf('当前金额: %d元', currentMoney);
            DrawFormattedText(win, moneyText, 'center', screenHeight * 0.8, [255 255 255]);
            Screen('Flip', win);
            
            % 等待按键
            [~, keyCode] = KbStrokeWait;
            
            if keyCode(KbName('space'))  % 充气
                % 检查是否爆炸
                if rand < currentProb
                    % 气球爆炸
                    exploded = true;
                    currentMoney = 0;
                    
                    % 显示爆炸动画
                    for i = 1:10
                        Screen('FillOval', win, [255 255 0], balloonRect);  % 黄色闪烁
                        Screen('Flip', win);
                        WaitSecs(0.05);
                        Screen('FillOval', win, [255 0 0], balloonRect);    % 红色闪烁
                        Screen('Flip', win);
                        WaitSecs(0.05);
                    end
                    
                    % 显示爆炸信息
                    DrawFormattedText(win, '气球爆炸了！\n\n按任意键继续', 'center', 'center', [255 255 255]);
                    Screen('Flip', win);
                    KbStrokeWait;
                else
                    % 气球变大
                    currentSize = min(currentSize + growthRate, maxSize);
                    currentMoney = currentMoney + moneyPerPump;
                    numPumps = numPumps + 1;
                    currentProb = currentProb + probIncrease;
                end
            elseif keyCode(KbName('return'))  % 结束当前trial
                break;
            end
        end
        
        % 记录数据
        data.numPumps(trial) = numPumps;
        data.earnings(trial) = currentMoney;
        data.explosions(trial) = exploded;
        
        % 显示本轮结果
        resultText = sprintf('本轮获得: %d元\n\n按任意键继续', currentMoney);
        DrawFormattedText(win, resultText, 'center', 'center', [255 255 255]);
        Screen('Flip', win);
        KbStrokeWait;
    end
    
    % 显示实验结束和总收益
    totalEarnings = sum(data.earnings);
    endText = sprintf('实验结束！\n\n总收益: %d元\n\n按任意键结束', totalEarnings);
    DrawFormattedText(win, endText, 'center', 'center', [255 255 255]);
    Screen('Flip', win);
    KbStrokeWait;
    
    % 关闭窗口
    Screen('CloseAll');
    
    % 保存数据
    save(['BART_' participantID '_session' num2str(session) '.mat'], 'data');
    
    disp(['BART task completed for participant ', participantID, ' in session ', num2str(session)]);
end

%% Emotion Task
function runEmotion(participantID, session)
    % 实验参数设置
    STIM_DURATION = 6;  % 图片呈现时间(秒)
    RATING_DURATION = 10;  % 评分时间(秒)
    
% 定义图片文件夹路径
STIM_DIR = struct();
STIM_DIR.positive = 'C:\Users\XX\Desktop\stimulus\positve';
STIM_DIR.neutral = 'C:\Users\XX\Desktop\stimulus\neutral';
STIM_DIR.negative = 'C:\Users\XX\Desktop\stimulus\negative';

% 初始化图片路径结构体
IMAGE_PATHS = struct();

% 从每个文件夹中读取图片
for category = {'positive', 'neutral', 'negative'}
cat = category{1};
% 获取文件夹中的所有jpg图片
files = dir(fullfile(STIM_DIR.(cat), '*.jpg'));

% 确保找到了图片
if isempty(files)
error('No jpg files found in folder: %s', STIM_DIR.(cat));
end

% 如果图片超过10张，随机选择10张
if length(files) > 10
selected_indices = randperm(length(files), 10);
files = files(selected_indices);
elseif length(files) < 10
warning('Less than 10 images found in %s folder', cat);
end

% 存储完整的文件路径
IMAGE_PATHS.(cat) = files;
end
    
    % 初始化Psychtoolbox
    Screen('Preference', 'SkipSyncTests', 1);
    [window, windowRect] = PsychImaging('OpenWindow', 0, [128 128 128]);
    Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');
    
    try
        % 显示实验说明
        showInstructions(window);
        
        % 基线阶段
        baseline_results = emotionRatingTask(window, IMAGE_PATHS, 'baseline');
        
        % 调节阶段（仅负性图片）
        regulation_results = regulationTask(window, IMAGE_PATHS.negative);
        
        % 保存结果
        all_results = [baseline_results; regulation_results];
        save('experiment_results.mat', 'all_results');
        
    catch err
        sca;
        rethrow(err);
    end
    
    % 关闭窗口
    sca;
end

function showInstructions(window)
    % 显示实验说明
    instruction_text = ['欢迎参加情绪实验。\n\n' ...
        '您将看到一系列图片，\n' ...
        '请根据您的感受对每张图片进行评分：\n\n' ...
        '效价评分：1(非常不愉快) 到 9(非常愉快)\n' ...
        '唤醒度评分：1(非常平静) 到 9(非常唤醒)\n\n' ...
        '按空格键继续'];
    
    DrawFormattedText(window, instruction_text, 'center', 'center', [255 255 255]);
    Screen('Flip', window);
    KbStrokeWait;
end

function results = emotionRatingTask(window, image_paths, phase)
    % 合并所有图片路径
    if strcmp(phase, 'baseline')
        all_images = [image_paths.positive; image_paths.neutral; image_paths.negative];
    else
        all_images = image_paths;
    end
    
    % 随机化图片顺序
    randomOrder = randperm(length(all_images));
    all_images = all_images(randomOrder);
    
    results = struct('image', {}, 'phase', {}, 'valence', {}, 'arousal', {});
    
    for i = 1:length(all_images)
        % 加载并显示图片
        img = imread(fullfile(all_images(i).folder, all_images(i).name));
        texture = Screen('MakeTexture', window, img);
        Screen('DrawTexture', window, texture);
        onset = Screen('Flip', window);
        
        % 等待图片呈现时间
        WaitSecs(STIM_DURATION);
        
        % 效价评分
        valence = getRating(window, '情绪效价评分', '1 = 非常不愉快, 9 = 非常愉快');
        
        % 唤醒度评分
        arousal = getRating(window, '唤醒度评分', '1 = 非常平静, 9 = 非常唤醒');
        
        % 保存结果
        results(i).image = all_images(i).name;
        results(i).phase = phase;
        results(i).valence = valence;
        results(i).arousal = arousal;
        
        Screen('Close', texture);
    end
end

function rating = getRating(window, title, instruction)
    % 创建评分界面
    DrawFormattedText(window, [title '\n\n' instruction], 'center', 'center', [255 255 255]);
    Screen('Flip', window);
    
    % 等待按键响应 (1-9)
    while true
        [~, keyCode] = KbStrokeWait;
        key = find(keyCode, 1);
        if any(key >= KbName('1') && key <= KbName('9'))
            rating = str2double(KbName(key));
            break;
        end
    end
end

function results = regulationTask(window, negative_images)
    % 显示调节任务说明
    regulation_text = ['接下来请尝试用不同角度重新解读图片内容。\n\n' ...
        '例如，将灾难场景想象为电影场景。\n\n' ...
        '按空格键继续'];
    
    DrawFormattedText(window, regulation_text, 'center', 'center', [255 255 255]);
    Screen('Flip', window);
    KbStrokeWait;
    
    % 执行调节任务评分
    results = emotionRatingTask(window, negative_images, 'regulation');
end

%% Helper Functions
% Stroop任务的辅助函数
function colorRGB = getColor(colorName)
    % 将颜色名称转换为RGB值
    % 输入参数：
    %   colorName - 颜色名称字符串 ('Red', 'Green', 或 'Blue')
    % 输出参数：
    %   colorRGB - RGB颜色值数组 [R G B]
    
    switch lower(colorName)
        case 'red'
            colorRGB = [255 0 0];      % 红色
        case 'green'
            colorRGB = [0 255 0];      % 绿色
        case 'blue'
            colorRGB = [0 0 255];      % 蓝色
        otherwise
            colorRGB = [255 255 255];  % 默认白色
    end
end

function waitForResponse()
    % waitForResponse函数代码
end

function [responseTime, isCorrect] = waitForResponseWithTiming(stimulusOnset, correctColor)
    % 等待按键并记录反应时和正确性
    while true
        [keyIsDown, secs, keyCode] = KbCheck;
        if keyIsDown
            responseTime = secs - stimulusOnset;
            pressedKey = KbName(keyCode);
            
            % 查是否是有效按键（1, 2, 或 3）
            if any(strcmpi(pressedKey, {'1!', '2@', '3#'}))
                isCorrect = checkResponse(pressedKey(1), correctColor);  % 只取第一个字符，忽略shift符号
                return;
            end
        end
    end
end

function isCorrect = checkResponse(responseKey, color)
    % 检查按键是否正确
    % responseKey: 被试按下的键（'1', '2', 或 '3'）
    % color: 正确的颜色名称（'Red', 'Green', 或 'Blue'）
    
    % 将按键和颜色对应
    switch responseKey
        case '1'
            isCorrect = strcmpi(color, 'Red');
        case '2'
            isCorrect = strcmpi(color, 'Green');
        case '3'
            isCorrect = strcmpi(color, 'Blue');
        otherwise
            isCorrect = false;
    end
end

% 添加练习trials的运行函数
function RunPracticeTrials(win, stimuli, targetPositions, stimDuration, isi, nback)
    for trial = 1:length(stimuli)
        % 显示刺激
        DrawFormattedText(win, stimuli{trial}, 'center', 'center', [255 255 255]);
        Screen('Flip', win);
        stimOnset = GetSecs;
        
        % 等待反应
        while GetSecs - stimOnset < stimDuration
            [keyIsDown, ~, keyCode] = KbCheck;
            if keyIsDown && keyCode(KbName('j'))
                if targetPositions(trial)
                    DrawFormattedText(win, '√', 'center', screenHeight * 0.8, [0 255 0]);
                else
                    DrawFormattedText(win, '×', 'center', screenHeight * 0.8, [255 0 0]);
                end
                Screen('Flip', win);
                break;
            end
        end
        
        % ISI
        WaitSecs(isi);
        Screen('Flip', win);
    end
end

function stimuli = GenerateStimuliSequence(letters, trialsPerBlock, nback)
    % 为N-back任务生成刺激序列
    % 输入参数：
    %   letters - 可用字母池（字符串）
    %   trialsPerBlock - 每个block的trial数量
    %   nback - N-back任务类型（0, 1, 或 2）
    % 输出参数：
    %   stimuli - 包含字母的cell数组
    
    % 初始化刺激序列
    stimuli = cell(1, trialsPerBlock);
    
    % 随机生成字母序列
    for i = 1:trialsPerBlock
        stimuli{i} = letters(randi(length(letters)));
    end
    
    % 对于n-back > 0的情况确保有足够的目标trials
    if nback > 0
        numTargets = floor(trialsPerBlock * 0.3);  % 设置30%的trials为目标
        % 随机选择目标位置（确保位置在nback之后）
        targetPositions = randperm(trialsPerBlock - nback, numTargets) + nback;
        
        % 在目标位置设置匹配的字母
        for pos = targetPositions
            stimuli{pos} = stimuli{pos - nback};  % 复制n个位置之前的字母
        end
    end
end

function [rt, accuracy] = RunTrials(win, stimuli, targetPositions, stimDuration, isi, nback)
    % 运行N-back任务的trials并记录反应时和正确率
    % 输出:
    %   rt - 反应时数组
    %   accuracy - 正确率数组
    
    rt = zeros(1, length(stimuli));        % 初始化反应时数组
    responses = zeros(1, length(stimuli));  % 初始化反应数组
    
    for trial = 1:length(stimuli)
        % 显示刺激
        DrawFormattedText(win, stimuli{trial}, 'center', 'center', [255 255 255]);
        Screen('Flip', win);
        stimOnset = GetSecs;
        responded = false;
        
        % 在刺激呈现期间等待反应
        while GetSecs - stimOnset < stimDuration
            [keyIsDown, secs, keyCode] = KbCheck;
            if keyIsDown && keyCode(KbName('j')) && ~responded
                rt(trial) = secs - stimOnset;
                responses(trial) = 1;
                responded = true;
                
                % 显示即时反馈（可选）
                if targetPositions(trial)
                    DrawFormattedText(win, '√', 'center', screenHeight * 0.8, [0 255 0]);
                else
                    DrawFormattedText(win, '×', 'center', screenHeight * 0.8, [255 0 0]);
                end
                Screen('Flip', win);
            end
        end
        
        % 如果没有反应，记录为0
        if ~responded
            rt(trial) = 0;
            responses(trial) = 0;
        end
        
        % ISI
        WaitSecs(isi);
        Screen('Flip', win);
    end
    
    % 计算正确率
    hits = sum(responses & targetPositions);  % 正确击中
    fas = sum(responses & ~targetPositions);  % 错误警报
    misses = sum(~responses & targetPositions);  % 漏报
    crs = sum(~responses & ~targetPositions);  % 正确拒绝
    
    % 计算整体正确率
    accuracy = (hits + crs) / length(stimuli);
    
    % 只保留有效反应的反应时（排除未反应的试次）
    rt = rt(rt > 0);
end


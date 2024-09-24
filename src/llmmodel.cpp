#include "llmmodel.h"
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QEventLoop>
#include <QDebug>
#include <QDateTime>
#include <QRegularExpression>

//bool containsMeaningfulCharacters(const QString &str) {
//    QRegularExpression re("(?![\\s\\p{P}]+$).*");
//    return re.match(str).hasMatch();
//}

bool containsMeaningfulCharacters(const QString &str) {
    QRegularExpression re("([\u4e00-\u9fa5]+|[a-zA-Z]+)");
    QRegularExpressionMatch match = re.match(str);
    if (match.hasMatch() && match.captured(0).length() > 1) {
        return true;
    }
    return false;
}
int findLastPunctuation(const QString &str, int iloc) {
    // 定义一个包含常见标点符号的字符串
    QString punctuations = "。，、；：？！“”‘’《》（）【】「」『』〈〉《》「」″‵′※‸※¤✲•·✦✧✵✲✳✱✾✿✺✹✷✸✾✽✼✻✾✿❀❁❂❃❇❈❉❊❋❌❍❎❏❒❖❗❘❙❚❛❜❝❞";
    int lastIndex = -1;
    int length = str.length();

    // 从字符串的末尾向前查找标点符号
    for (int i = length - 1; i >= iloc; --i) {
        if (punctuations.contains(str[i])) {
            lastIndex = i;
            break;
        }
    }

    return lastIndex;
}

LLMModel::LLMModel()
{
//    connect(m_timer, &QTimer::timeout, this, &LLMModel::Update);
}
LLMModel::~LLMModel()
{

}
void LLMModel::abort()
{
    m_networkReply->abort();
    m_llm_text.clear();
}
void LLMModel::Reset()
{
}

void LLMModel::Run(QString text, int port_num)
{
    m_loc_LastPunctuation = 0;
    m_loc_NextPunctuation = 0;
    m_llm_text.clear();
    m_networkAccessManager = new QNetworkAccessManager(this);
    QNetworkRequest request;
    QString url = "http://localhost:" + QString::number(port_num) + "/v1/chat/completions";
    qDebug() << url;
    request.setUrl(QUrl(url));
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    // 创建消息数组并添加上面创建的QJsonObject
    QJsonArray messagesArray;
    // 创建包含消息的QJsonObject
    QJsonObject messageObject;
    messageObject["role"] = "system";
    messageObject["content"] = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
    messagesArray.append(messageObject);
    messageObject["role"] = "user";
    messageObject["content"] = text;
    messagesArray.append(messageObject);

    // 创建顶层的QJsonObject并添加消息数组和stream键
    QJsonObject rootObject;
//    rootObject["model"] = "Qwen/Qwen2.5-3B-Instruct";
    rootObject["model"] = "Qwen/Qwen2.5-Instruct";
    rootObject["messages"] = messagesArray;
    rootObject["stream"] = true;
    rootObject["temperature"] = 0.7;
    rootObject["top_p"] = 0.8;
    rootObject["max_tokens"] = 2048;

    QJsonDocument jsonDoc(rootObject);
    qDebug() << "QJsonDocument" << jsonDoc << text;
    QByteArray jsonData = jsonDoc.toJson();
    m_networkReply = m_networkAccessManager->post(request, jsonData);
    connect(m_networkReply, &QNetworkReply::readyRead, [&]() {
        // 当有数据可读时，进行处理
        QByteArray data = m_networkReply->readAll();
        QString allData(data);
        if(allData.startsWith("data: ")){
            QStringList msgs=allData.split("data: ");
//            qDebug() << "SSSSSSSSSS" << msgs.length() << msgs;

            // 类似"data: {\"choices\":[{\"finish_reason\":\"stop\",\"index\":0,\"delta\":{}}],\"created\":1727019183,\"id\":\"chatcmpl-WSasYf1ZCvCg8Co6kxtmCbJBV9V2CXhQ\",\"model\":\"Qwen/Qwen2.5-3B-Instruct\",\"object\":\"chat.completion.chunk\",\"usage\":{\"completion_tokens\":9,\"prompt_tokens\":31,\"total_tokens\":40}}\n\ndata: [DONE]\n\n"
            // 一般一个"data:"，如果有第二个，一般是"[DONE]\n\n"代表结束，但有时也会有第二个常规数据
            for (auto elem: msgs)
            {
                if (elem == "")
                {
                    // 什么都不做
                }
                else if (elem == "[DONE]\n\n")
                {
                    qDebug() << "SSSSSSSSSS" << msgs.length() << msgs;
                    m_llm_text.is_endpoint = true;
                    QString new_str = m_llm_text.text.mid(m_loc_LastPunctuation, m_llm_text.text.length() - m_loc_LastPunctuation);
                    if (containsMeaningfulCharacters(new_str))
                    {
                        m_llm_text.sub_sentences.append(new_str);
                    }
                }
                else
                {
                    QString elem = msgs.at(1);
                    QJsonDocument doc=QJsonDocument::fromJson(elem.toUtf8());
                    QJsonObject obj=doc.object();
                    QJsonArray arr=obj["choices"].toArray();
                    if(arr.size()>0){
                        QJsonObject ja_choice = arr[0].toObject();
                        QJsonObject jo_delta = ja_choice["delta"].toObject();
                        QString delta_text = jo_delta["content"].toString();
                        m_llm_text.tokens.append(delta_text);
                        m_llm_text.text += delta_text;
                    }
                    m_loc_NextPunctuation = findLastPunctuation(m_llm_text.text, m_loc_LastPunctuation) + 1;
                    if (m_loc_NextPunctuation - m_loc_LastPunctuation > 10)
                    {
                        m_llm_text.sub_sentences.append(m_llm_text.text.mid(m_loc_LastPunctuation, m_loc_NextPunctuation - m_loc_LastPunctuation));
                        m_loc_LastPunctuation = m_loc_NextPunctuation;
                    }
                }

            }
            emit SignalNewAnswer();
        }
    });
//    connect(m_networkReply, &QNetworkReply::finished, m_networkReply, &QNetworkReply::deleteLater);
}

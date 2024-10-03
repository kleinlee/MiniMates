#ifndef LLMMODEL_H
#define LLMMODEL_H

#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QObject>
#include <QString>
#include <QObject>
#include <QTimer>
struct LlmText
{
    QString text;
    QList<QString> sub_sentences;
    QList<QString> tokens;
    bool is_endpoint = false;
    // 清空数据的成员函数
    void clear() {
        text.clear();
        sub_sentences.clear();
        tokens.clear();
        is_endpoint = false;
    }
};
class LLMModel : public QObject
{
    Q_OBJECT
public:
    LLMModel();
    ~LLMModel();

    void Run(QString text, int port_num);
    void abort();

    LlmText m_llm_text;
    int m_loc_LastPunctuation = 0;
    int m_loc_NextPunctuation = 0;

    QNetworkAccessManager *m_networkAccessManager = nullptr;
    QNetworkReply *m_networkReply = nullptr;

    std::string m_input;
    QTimer* m_timer;

    QStringList m_output;
signals:
    void SignalNewAnswer();
public slots:
    void Reset();
};

#endif // LLMMODEL_H

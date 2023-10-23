import json
import numpy as np
from utils import get_bedrock_client
from typing import List

FLAGXS_SALES_POINTS = [
    "バラバラなExcelやツールを駆使したプロジェクトマネジメントから脱却し、マネジメント要素がシームレスに繋がった一元的で整合性のとれたマネジメントプラットフォームを提供します。",
    "Flagxsの利用と共にプロセスを整理することで、PDRI＝Plan→Do→Report→Decision/Improveのサイクルを定着化します。それにより常に進捗状況を定量的に確認できる状態になり、意思決定と改善に注力したプロジェクト管理を実現します。",
    "これまで属人的に階層化されていたWBSを工程・成果物・タスクに分解しツリー構造で管理します。「型」にはめることでWBSの標準化が進み、マネージャーから現場までの目線のあったプロジェクト管理を実現します。",
    "プロジェクト計画の初期段階は「工程・成果物レベルで見積り」、工程の開始前に「タスクレベルで見積り」、作業実行において「タスクレベルに投入工数を記録」することで、解像度の高い見積りと実績を確認することができます。見積りと実績を比較することにより生産性を精度高く確認することが可能です。",
    "計画を立てる際はExcelライクに入力のストレスなく作業することができます。また、数クリックで実績登録→定量レポートを確認することができ、現場に負担がかからないUI/UXを実現しています。",
    "作業実績入力と同時に進捗状況がリアルタイム集計され、いつでも最新のレポートを確認することができます。これまで実績集計とレポート作成に費やしていた工数を大幅に削減し、リアルタイムな状況判断を実現します。",
    "Flagxsはプロジェクト管理に必要な各種進捗レポートを標準搭載しています。進捗報告レポートの設計やExcelダウンロード・マクロ集計といった作業をすることなく、各種進捗レポートをいつでも最新の情報で出力することが可能です。",
    "FlagxsはWBSに議論の経緯や決定事項、成果物を保存することが可能です。それらをWBSに記録することでテーマも場所も明確になります。また、標準化されたWBSツリー構造と通知機能で対象WBSを探しやすく過去経緯の振り返りが容易です。",
    "これまでプロジェクト毎の使い切りだったプロジェクト実績情報が、次のプロジェクトに活かせるナレッジ＝情報資産となります。また、SaaSアプリケーションであるFlagxsが継続的に進化することで、蓄積したナレッジの活用と共により高度なマネジメントが可能となります。",
    "経験豊かなFlagxs導入コンサルティングの専門家が、Flagxsの操作説明・オリエンテーションからプロジェクト管理のプロセスの定着化・改善を支援、プロジェクトの成功に向け伴走します。",
    "リモートワークやオフショアでの活用を前提としたクラウドプラットフォームです。世界中どこにいても使用することができます。多言語対応も進めています（現在一部機能）",
    "弊社クラウドを利用することが前提となります。セキュリティ強化として、Azure認証の利用を推奨しています。",
    "スマートフォンは今後対応する予定です。現在は、Chromeをサポートブラウザとしております。",
    "ユーザー毎に、アクセス可能なメニューとプロジェクトに対するログイン権限を設定することが可能です。",
    "ログインログとアプリケーションアクセスログを管理しています。バックアップはデイリーで実施しており、災害時は前日までのデータの復旧が可能です。",
    "導入コンサルティングサービスを利用することで、単なるツール提供ではなく経験豊富なメンバーがチームやメンバーが自立するまでサポートします。",
    """
    コスト効果
    - プロジェクト管理においての業務効率化
    - チーム毎に存在する異なる粒度のExcelのWBSを統合して集計・レポートする
    - 課題管理表やRedmine等にバラバラに存在するWBS外作業を集計・レポートする
    - WBSの日々の実績計上や集計・レポートにおける不整合対応
    - 遅延タスクを抽出して追い込みをかける
    - 意図したマネジメントを維持するためのコスト
    - 共有設定されたExcel WBSが破損した際の対応
    """,
    """
    付加価値
    - これまで実現できなかったプロジェクト管理
    - いつでも、だれでもリアルタイムに進捗状況が見れる
    - 複数プロジェクトに跨った進捗状況をApple to Appleで把握できる
    - 工程毎、チーム毎、メンバー毎の定量的な進捗状況や遅延をドリルダウンで把握できる
    - リアルタイムに計画差異を確認し、着地見込みを工数、期日の両面で確認することができる
    - 集計・レポート業務から開放され、本来するべき仕事にマネージャーが集中できる
    """,
]

client = get_bedrock_client()


def get_embedding(text: str) -> np.ndarray:
    body = json.dumps({"inputText": text})

    modelId = "amazon.titan-embed-text-v1"
    accept = "*/*"
    contentType = "application/json"

    response = client.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())
    # print(response)
    response_embeddings = np.array(response_body.get("embedding"))

    return response_embeddings


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def search_embeddings(query: str, threshold=0.5) -> str:
    query_embedding = get_embedding(query)
    cos_sims = []

    # リストの中の埋め込み表現とのコサイン類似度を計算
    for embd in [get_embedding(e) for e in FLAGXS_SALES_POINTS]:
        cos_sims.append(cos_sim(embd, query_embedding))

    # 類似度の降順にソート
    decorated = [
        (cos_sims[i], i, element) for i, element in enumerate(FLAGXS_SALES_POINTS)
    ]
    decorated.sort(reverse=True)

    # 類似度が一定のしきい値を超えているエントリのみを返す
    result = [
        (cos_sim, element) for cos_sim, _, element in decorated if cos_sim > threshold
    ]

    return result[0][1]

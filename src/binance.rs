use anyhow::Result;
use base64::Engine as _;
use chrono::Utc;
use hmac::{Hmac, Mac};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::HashMap;
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone)]
pub struct BinanceClient {
    pub client: Client,
    api_key: String,
    secret_key: String,
    pub base_url: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OrderBookTicker {
    pub symbol: String,
    #[serde(rename = "bidPrice")]
    pub bid_price: String,
    #[serde(rename = "bidQty")]
    pub bid_qty: String,
    #[serde(rename = "askPrice")]
    pub ask_price: String,
    #[serde(rename = "askQty")]
    pub ask_qty: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SymbolInfo {
    pub symbol: String,
    pub status: String,
    #[serde(rename = "baseAsset")]
    pub base_asset: String,
    #[serde(rename = "quoteAsset")]
    pub quote_asset: String,
    pub filters: Vec<SymbolFilter>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SymbolFilter {
    #[serde(rename = "filterType")]
    pub filter_type: String,
    #[serde(rename = "minPrice")]
    pub min_price: Option<String>,
    #[serde(rename = "maxPrice")]
    pub max_price: Option<String>,
    #[serde(rename = "tickSize")]
    pub tick_size: Option<String>,
    #[serde(rename = "minQty")]
    pub min_qty: Option<String>,
    #[serde(rename = "maxQty")]
    pub max_qty: Option<String>,
    #[serde(rename = "stepSize")]
    pub step_size: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AccountInfo {
    #[serde(rename = "makerCommission")]
    pub maker_commission: u32,
    #[serde(rename = "takerCommission")]
    pub taker_commission: u32,
    #[serde(rename = "buyerCommission")]
    pub buyer_commission: u32,
    #[serde(rename = "sellerCommission")]
    pub seller_commission: u32,
    #[serde(rename = "canTrade")]
    pub can_trade: bool,
    #[serde(rename = "canWithdraw")]
    pub can_withdraw: bool,
    #[serde(rename = "canDeposit")]
    pub can_deposit: bool,
    pub balances: Vec<Balance>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Balance {
    pub asset: String,
    pub free: String,
    pub locked: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct NewOrderResponse {
    pub symbol: String,
    #[serde(rename = "orderId")]
    pub order_id: u64,
    #[serde(rename = "orderListId")]
    pub order_list_id: i64,
    #[serde(rename = "clientOrderId")]
    pub client_order_id: String,
    #[serde(rename = "transactTime")]
    pub transact_time: u64,
    pub price: String,
    #[serde(rename = "origQty")]
    pub orig_qty: String,
    #[serde(rename = "executedQty")]
    pub executed_qty: String,
    #[serde(rename = "cummulativeQuoteQty")]
    pub cummulative_quote_qty: String,
    pub status: String,
    #[serde(rename = "timeInForce")]
    pub time_in_force: String,
    #[serde(rename = "type")]
    pub order_type: String,
    pub side: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Kline {
    pub open_time: u64,
    pub open: String,
    pub high: String,
    pub low: String,
    pub close: String,
    pub volume: String,
    pub close_time: u64,
    pub quote_asset_volume: String,
    pub number_of_trades: u32,
    pub taker_buy_base_asset_volume: String,
    pub taker_buy_quote_asset_volume: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Ticker24hr {
    pub symbol: String,
    #[serde(rename = "priceChange")]
    pub price_change: String,
    #[serde(rename = "priceChangePercent")]
    pub price_change_percent: String,
    #[serde(rename = "weightedAvgPrice")]
    pub weighted_avg_price: String,
    #[serde(rename = "prevClosePrice")]
    pub prev_close_price: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "lastQty")]
    pub last_qty: String,
    #[serde(rename = "bidPrice")]
    pub bid_price: String,
    #[serde(rename = "askPrice")]
    pub ask_price: String,
    #[serde(rename = "openPrice")]
    pub open_price: String,
    #[serde(rename = "highPrice")]
    pub high_price: String,
    #[serde(rename = "lowPrice")]
    pub low_price: String,
    pub volume: String,
    #[serde(rename = "quoteVolume")]
    pub quote_volume: String,
    #[serde(rename = "openTime")]
    pub open_time: u64,
    #[serde(rename = "closeTime")]
    pub close_time: u64,
    #[serde(rename = "firstId")]
    pub first_id: u64,
    #[serde(rename = "lastId")]
    pub last_id: u64,
    pub count: u32,
}

impl BinanceClient {
    pub fn new(api_key: String, secret_key: String, base_url: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            secret_key,
            base_url,
        }
    }

    fn get_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    fn sign(&self, query_string: &str) -> String {
        let mut mac = HmacSha256::new_from_slice(self.secret_key.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(query_string.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    fn build_query_string(&self, params: &HashMap<String, String>) -> String {
        let mut query_params = params.clone();
        query_params.insert("timestamp".to_string(), Self::get_timestamp().to_string());

        let query_string = query_params
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("&");

        let signature = self.sign(&query_string);
        format!("{query_string}&signature={signature}")
    }

    pub async fn get_account_info(&self) -> Result<AccountInfo> {
        let params = HashMap::new();
        let query_string = self.build_query_string(&params);

        let url = format!("{}/api/v3/account?{}", self.base_url, query_string);

        let response = self
            .client
            .get(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await?
            .json::<AccountInfo>()
            .await?;

        Ok(response)
    }

    pub async fn get_order_book_ticker(&self, symbol: &str) -> Result<OrderBookTicker> {
        let url = format!(
            "{}/api/v3/ticker/bookTicker?symbol={}",
            self.base_url, symbol
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await?
            .json::<OrderBookTicker>()
            .await?;

        Ok(response)
    }

    pub async fn get_24hr_ticker(&self, symbol: &str) -> Result<Ticker24hr> {
        let url = format!("{}/api/v3/ticker/24hr?symbol={}", self.base_url, symbol);

        let response = self
            .client
            .get(&url)
            .send()
            .await?
            .json::<Ticker24hr>()
            .await?;

        Ok(response)
    }

    pub async fn get_klines(&self, symbol: &str, interval: &str, limit: u16) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/api/v3/klines?symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: Vec<Vec<serde_json::Value>> =
            self.client.get(&url).send().await?.json().await?;

        let klines = response
            .into_iter()
            .map(|k| Kline {
                open_time: k[0].as_u64().unwrap(),
                open: k[1].as_str().unwrap().to_string(),
                high: k[2].as_str().unwrap().to_string(),
                low: k[3].as_str().unwrap().to_string(),
                close: k[4].as_str().unwrap().to_string(),
                volume: k[5].as_str().unwrap().to_string(),
                close_time: k[6].as_u64().unwrap(),
                quote_asset_volume: k[7].as_str().unwrap().to_string(),
                number_of_trades: k[8].as_u64().unwrap() as u32,
                taker_buy_base_asset_volume: k[9].as_str().unwrap().to_string(),
                taker_buy_quote_asset_volume: k[10].as_str().unwrap().to_string(),
            })
            .collect();

        Ok(klines)
    }

    pub async fn place_market_order(
        &self,
        symbol: &str,
        side: &str,
        quantity: &str,
    ) -> Result<NewOrderResponse> {
        let quantity = format!(
            "{:.5}",
            Decimal::from_str(quantity).unwrap_or(Decimal::ZERO)
        );
        let mut params = HashMap::new();
        params.insert("symbol".to_string(), symbol.to_string());
        params.insert("side".to_string(), side.to_string());
        params.insert("type".to_string(), "MARKET".to_string());
        params.insert("quantity".to_string(), quantity.to_string());

        let query_string = self.build_query_string(&params);
        let url = format!("{}/api/v3/order", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(query_string)
            .send()
            .await?
            .json::<NewOrderResponse>()
            .await?;

        Ok(response)
    }

    pub async fn place_limit_order(
        &self,
        symbol: &str,
        side: &str,
        quantity: &str,
        price: &str,
        time_in_force: &str,
    ) -> Result<NewOrderResponse> {
        // remove the last two characters from the price regardless of the input
        let price = format!("{:.2}", Decimal::from_str(price).unwrap_or(Decimal::ZERO));
        let quantity = format!(
            "{:.5}",
            Decimal::from_str(quantity).unwrap_or(Decimal::ZERO)
        );

        let mut params = HashMap::new();
        params.insert("symbol".to_string(), symbol.to_string());
        params.insert("side".to_string(), side.to_string());
        params.insert("type".to_string(), "LIMIT".to_string());
        params.insert("timeInForce".to_string(), time_in_force.to_string());
        params.insert("quantity".to_string(), quantity.to_string());
        params.insert("price".to_string(), price.to_string());
        params.insert("recvWindow".to_string(), "5000".to_string());

        // Required by Binance
        let timestamp = Utc::now().timestamp_millis().to_string();
        params.insert("timestamp".to_string(), timestamp);

        let query_string = self.build_query_string(&params);

        let url = format!("{}/api/v3/order", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(query_string)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Error placing order: {}", error_text));
        }

        let response = response.json::<NewOrderResponse>().await?;

        Ok(response)
    }
}

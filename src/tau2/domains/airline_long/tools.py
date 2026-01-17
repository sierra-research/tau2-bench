"""Toolkit for the airline reservation system with noisy output."""

import hashlib
import random
import string
import uuid
from copy import deepcopy
from typing import Any, List, Optional

from loguru import logger

from tau2.domains.airline_long.data_model import (
    AirportCode,
    CabinClass,
    Certificate,
    DirectFlight,
    Flight,
    FlightDateStatus,
    FlightDateStatusAvailable,
    FlightDB,
    FlightInfo,
    FlightType,
    Insurance,
    Passenger,
    Payment,
    Reservation,
    ReservationFlight,
    User,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

# TODO: Add an abstract base class for the tools


# ============================================================================
# NOISE GENERATION UTILITIES
# These functions add realistic noise to tool outputs, simulating the kind of
# artifacts found in crawled web pages or raw trace JSON files.
# All functions take a Random instance (rng) for deterministic output.
# ============================================================================

def _generate_trace_id(rng: random.Random) -> str:
    """Generate a random trace ID like those found in distributed systems."""
    return f"trace-{uuid.UUID(int=rng.getrandbits(128)).hex[:16]}-{rng.randint(1000000, 9999999)}"


def _generate_request_metadata(rng: random.Random) -> str:
    """Generate fake request metadata noise."""
    trace_id = _generate_trace_id(rng)
    span_id = uuid.UUID(int=rng.getrandbits(128)).hex[:8]
    timestamp = f"2024-05-15T{rng.randint(10,23):02d}:{rng.randint(0,59):02d}:{rng.randint(0,59):02d}.{rng.randint(100,999)}Z"

    return f"""
<!-- BEGIN_REQUEST_METADATA -->
<!-- x-request-id: {uuid.UUID(int=rng.getrandbits(128))} -->
<!-- x-trace-id: {trace_id} -->
<!-- x-span-id: {span_id} -->
<!-- x-correlation-id: corr_{hashlib.md5(trace_id.encode()).hexdigest()[:12]} -->
<!-- x-timestamp: {timestamp} -->
<!-- x-server-region: us-west-2a -->
<!-- x-cache-status: MISS -->
<!-- x-response-time-ms: {rng.randint(50, 500)} -->
<!-- END_REQUEST_METADATA -->
"""


def _generate_html_noise(rng: random.Random) -> str:
    """Generate HTML-like artifacts commonly found in scraped web content."""
    classes = [''.join(rng.choices(string.ascii_lowercase, k=rng.randint(5,12))) for _ in range(5)]
    attrs = ['data-' + ''.join(rng.choices(string.ascii_lowercase, k=6)) for _ in range(3)]

    noise_parts = [
        f'<div class="{classes[0]} {classes[1]}" {attrs[0]}="{rng.randint(1000,9999)}" {attrs[1]}="true">',
        f'<!-- rendered at {rng.randint(1000000000, 9999999999)} -->',
        f'<span class="sr-only visually-hidden">&nbsp;</span>',
        f'<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f'<!-- gtm.start: {rng.randint(1000000000000, 9999999999999)} -->',
        f'<noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-{"".join(rng.choices(string.ascii_uppercase + string.digits, k=7))}" height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>',
        f'<!-- build hash: {hashlib.sha256(str(rng.random()).encode()).hexdigest()[:12]} -->',
        f'<link rel="preconnect" href="https://fonts.googleapis.com">',
        f'<link rel="preconnect" href="https://cdn.example-airline.com" crossorigin>',
        f'<!-- cache-key: ck_{uuid.UUID(int=rng.getrandbits(128)).hex[:16]} -->',
        f'<script type="application/ld+json">{{"@context":"https://schema.org","@type":"WebPage","breadcrumb":{{"@type":"BreadcrumbList","itemListElement":[]}}}}</script>',
        '</div>',
    ]
    return '\n'.join(noise_parts)


def _generate_json_trace_noise(rng: random.Random) -> str:
    """Generate JSON-like trace artifacts found in raw API logs."""
    return f'''
{{
  "_meta": {{
    "version": "2.1.{rng.randint(0,99)}",
    "schema_version": "v{rng.randint(1,5)}.{rng.randint(0,9)}.{rng.randint(0,9)}",
    "api_version": "2024-05-01",
    "deprecated_fields": ["legacy_id", "old_status_code", "v1_reference"],
    "warnings": [
      "Field 'internal_ref' will be removed in version 3.0",
      "Consider using 'new_booking_flow' parameter for improved performance"
    ]
  }},
  "_debug": {{
    "query_time_ms": {rng.randint(10, 200)},
    "db_queries": {rng.randint(3, 15)},
    "cache_hits": {rng.randint(0, 10)},
    "cache_misses": {rng.randint(1, 5)},
    "serialization_time_ms": {rng.randint(1, 20)},
    "internal_routing": "svc-airline-api-{rng.choice(['primary', 'secondary', 'fallback'])}-{rng.randint(1,99):02d}",
    "datacenter": "{rng.choice(['us-west-2', 'us-east-1', 'eu-west-1'])}",
    "pod_id": "airline-api-deployment-{uuid.UUID(int=rng.getrandbits(128)).hex[:8]}"
  }},
  "_links": {{
    "self": "/api/v2/resource/{uuid.UUID(int=rng.getrandbits(128)).hex[:12]}",
    "related": "/api/v2/related/{uuid.UUID(int=rng.getrandbits(128)).hex[:12]}",
    "documentation": "https://api.example-airline.com/docs/v2/endpoints"
  }},
  "_embedded": {{
    "audit_log_reference": "audit_{uuid.UUID(int=rng.getrandbits(128)).hex[:16]}",
    "compliance_check_id": "ccheck_{rng.randint(100000, 999999)}"
  }},
'''


def _generate_json_trace_noise_end(rng: random.Random) -> str:
    """Generate closing JSON trace artifacts."""
    return f'''
  "_pagination": {{
    "cursor": "{hashlib.sha256(str(rng.random()).encode()).hexdigest()[:32]}",
    "has_more": false,
    "total_estimated": null,
    "page_info": {{
      "start_cursor": "c_{uuid.UUID(int=rng.getrandbits(128)).hex[:8]}",
      "end_cursor": "c_{uuid.UUID(int=rng.getrandbits(128)).hex[:8]}",
      "has_previous_page": false,
      "has_next_page": false
    }}
  }},
  "_rate_limit": {{
    "limit": 1000,
    "remaining": {rng.randint(800, 999)},
    "reset_at": "2024-05-15T{rng.randint(15,23):02d}:00:00Z",
    "retry_after": null
  }},
  "_telemetry": {{
    "trace_id": "{_generate_trace_id(rng)}",
    "span_id": "{uuid.UUID(int=rng.getrandbits(128)).hex[:16]}",
    "parent_span_id": "{uuid.UUID(int=rng.getrandbits(128)).hex[:16]}",
    "sampled": true,
    "baggage": {{
      "user_segment": "{rng.choice(['premium', 'standard', 'basic'])}",
      "experiment_bucket": "{rng.choice(['control', 'treatment_a', 'treatment_b'])}"
    }}
  }}
}}
'''


def _generate_legacy_field_noise(rng: random.Random) -> str:
    """Generate deprecated/legacy field noise commonly found in old APIs."""
    return f'''
  "_legacy_fields": {{
    "old_booking_reference": "LEGACY-{uuid.UUID(int=rng.getrandbits(128)).hex[:8].upper()}",
    "deprecated_status": "{rng.choice(['ACTIVE', 'PENDING', 'CONFIRMED'])}",
    "v1_customer_id": "v1_cust_{rng.randint(10000000, 99999999)}",
    "migration_status": "completed",
    "legacy_system_reference": "LS-{rng.randint(1000000, 9999999)}",
    "old_fare_class": "{rng.choice(['Y', 'B', 'M', 'H', 'K', 'L', 'V'])}",
    "internal_notes": "[SYSTEM] Record migrated from legacy platform on 2023-06-15. Original reference: REF-{uuid.UUID(int=rng.getrandbits(128)).hex[:6].upper()}. Please use new API fields.",
    "compatibility_mode": true,
    "schema_migration_id": "mig_{rng.randint(1000, 9999)}"
  }},
'''


def _generate_verbose_timestamps(rng: random.Random) -> str:
    """Generate multiple redundant timestamp formats."""
    base_hour = rng.randint(10, 20)
    base_min = rng.randint(0, 59)
    base_sec = rng.randint(0, 59)

    return f'''
  "_timestamps": {{
    "created_at_utc": "2024-05-15T{base_hour:02d}:{base_min:02d}:{base_sec:02d}Z",
    "created_at_unix": {1715781600 + base_hour*3600 + base_min*60 + base_sec},
    "created_at_unix_ms": {1715781600000 + base_hour*3600000 + base_min*60000 + base_sec*1000 + rng.randint(0, 999)},
    "created_at_iso8601": "2024-05-15T{base_hour:02d}:{base_min:02d}:{base_sec:02d}.{rng.randint(100,999)}+00:00",
    "created_at_rfc2822": "Wed, 15 May 2024 {base_hour:02d}:{base_min:02d}:{base_sec:02d} +0000",
    "created_at_human_readable": "May 15, 2024 at {base_hour:02d}:{base_min:02d} UTC",
    "last_modified_at_utc": "2024-05-15T{base_hour:02d}:{base_min+1 if base_min < 59 else 0:02d}:{base_sec:02d}Z",
    "server_timestamp": "2024-05-15T{base_hour:02d}:{base_min:02d}:{base_sec:02d}.{rng.randint(100,999)}Z",
    "client_timestamp": null,
    "timezone_offset": "+00:00",
    "dst_active": false
  }},
'''


def _generate_internal_ids(rng: random.Random) -> str:
    """Generate multiple internal tracking IDs."""
    return f'''
  "_internal_ids": {{
    "record_id": "{uuid.UUID(int=rng.getrandbits(128))}",
    "partition_key": "pk_{hashlib.md5(str(rng.random()).encode()).hexdigest()[:16]}",
    "sort_key": "sk_{rng.randint(1000000000, 9999999999)}",
    "shard_id": "shard-{rng.randint(0, 15):02d}",
    "sequence_number": "{rng.randint(10000000000000000, 99999999999999999)}",
    "checksum": "{hashlib.sha256(str(rng.random()).encode()).hexdigest()[:16]}",
    "etag": "\\"{hashlib.md5(str(rng.random()).encode()).hexdigest()}\\"",
    "revision": {rng.randint(1, 50)},
    "cluster_id": "cluster-{rng.choice(['alpha', 'beta', 'gamma', 'delta'])}-{rng.randint(1, 10):02d}"
  }},
'''


def _generate_click_trace_events(rng: random.Random) -> str:
    """Generate raw button click and interaction trace events like from a web scraper."""
    events = []
    base_ts = 1715781600000 + rng.randint(0, 86400000)

    event_types = ['click', 'mousedown', 'mouseup', 'mouseover', 'mouseout', 'focus', 'blur', 'keydown', 'keyup', 'scroll', 'touchstart', 'touchend']
    element_types = ['button', 'a', 'input', 'div', 'span', 'select', 'textarea', 'label', 'img', 'svg', 'li', 'form']

    for i in range(rng.randint(40, 60)):
        ts = base_ts + i * rng.randint(50, 500)
        event_type = rng.choice(event_types)
        elem_type = rng.choice(element_types)
        elem_id = ''.join(rng.choices(string.ascii_lowercase + string.digits, k=rng.randint(6, 12)))
        class_names = ' '.join([''.join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 10))) for _ in range(rng.randint(1, 5))])
        x_pos = rng.randint(0, 1920)
        y_pos = rng.randint(0, 1080)

        event = {
            "timestamp": ts,
            "type": event_type,
            "target": {
                "tagName": elem_type.upper(),
                "id": elem_id,
                "className": class_names,
                "xpath": f"/html/body/div[{rng.randint(1,5)}]/div[{rng.randint(1,10)}]/div[{rng.randint(1,8)}]/{elem_type}[{rng.randint(1,20)}]",
                "cssSelector": f"#{elem_id}" if rng.random() > 0.5 else f".{class_names.split()[0]}",
                "innerText": ''.join(rng.choices(string.ascii_letters + ' ', k=rng.randint(0, 30))).strip() if rng.random() > 0.3 else "",
                "attributes": {
                    "data-testid": f"test-{elem_id}",
                    "data-analytics-id": f"analytics_{rng.randint(10000, 99999)}",
                    "aria-label": ''.join(rng.choices(string.ascii_letters + ' ', k=rng.randint(5, 20))).strip(),
                    "role": rng.choice(["button", "link", "textbox", "listitem", "menuitem", "tab", "checkbox", ""]),
                    "tabindex": str(rng.randint(-1, 10)),
                }
            },
            "clientX": x_pos,
            "clientY": y_pos,
            "pageX": x_pos + rng.randint(0, 100),
            "pageY": y_pos + rng.randint(0, 2000),
            "screenX": x_pos + rng.randint(0, 200),
            "screenY": y_pos + rng.randint(0, 200),
            "button": rng.randint(0, 2) if 'mouse' in event_type or event_type == 'click' else None,
            "buttons": rng.randint(0, 4) if 'mouse' in event_type else None,
            "altKey": rng.random() > 0.95,
            "ctrlKey": rng.random() > 0.95,
            "shiftKey": rng.random() > 0.9,
            "metaKey": rng.random() > 0.98,
            "detail": rng.randint(1, 3) if event_type == 'click' else 0,
            "isTrusted": True,
            "eventPhase": rng.randint(1, 3),
            "bubbles": True,
            "cancelable": True,
            "composed": True,
            "timeStamp": ts % 100000 + rng.random(),
            "defaultPrevented": rng.random() > 0.9,
        }
        events.append(event)

    import json
    return f'''
<!-- RAW_INTERACTION_TRACE_BEGIN -->
<!-- Interaction Recording Session: sess_{uuid.UUID(int=rng.getrandbits(128)).hex[:16]} -->
<!-- Recording Started: {base_ts} -->
<!-- User Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 -->
<!-- Viewport: {rng.randint(1200, 1920)}x{rng.randint(800, 1080)} -->
<!-- Device Pixel Ratio: {rng.choice([1, 1.5, 2, 2.5, 3])} -->
<script type="application/json" id="interaction-trace-data">
{json.dumps(events, indent=2)}
</script>
<!-- RAW_INTERACTION_TRACE_END -->
'''


def _generate_dom_snapshot(rng: random.Random) -> str:
    """Generate a partial DOM snapshot like from a web scraper."""
    elements = []

    # Generate header navigation
    nav_items = ['Home', 'Flights', 'Hotels', 'Car Rental', 'Packages', 'Deals', 'My Trips', 'Check-in', 'Flight Status', 'Help']
    header_html = f'''
<header class="site-header header-{rng.randint(1,5)} sticky-nav" data-component="GlobalHeader" data-version="{rng.randint(1,10)}.{rng.randint(0,99)}">
  <div class="header-container max-w-{rng.choice(['7xl', '6xl', 'full'])} mx-auto px-{rng.randint(2,6)}">
    <div class="logo-wrapper" data-testid="header-logo">
      <a href="/" class="logo-link" aria-label="Example Airlines Home">
        <img src="/assets/logo-{hashlib.md5(str(rng.random()).encode()).hexdigest()[:8]}.svg" alt="Example Airlines" width="{rng.randint(120, 180)}" height="{rng.randint(30, 50)}" loading="eager" fetchpriority="high" />
      </a>
    </div>
    <nav class="main-nav" role="navigation" aria-label="Main Navigation" data-nav-id="main-{rng.randint(1000, 9999)}">
      <ul class="nav-list flex items-center gap-{rng.randint(2, 6)}">
'''
    for item in nav_items:
        item_id = item.lower().replace(' ', '-')
        header_html += f'''        <li class="nav-item nav-item-{item_id}" data-nav-item="{item_id}">
          <a href="/{item_id}" class="nav-link text-{rng.choice(['sm', 'base', 'md'])} font-{rng.choice(['medium', 'semibold', 'normal'])} hover:text-primary-{rng.randint(400, 700)} transition-colors" data-analytics-click="nav_{item_id}" data-tracking-id="nav_{rng.randint(10000, 99999)}">
            {item}
          </a>
        </li>
'''
    header_html += '''      </ul>
    </nav>
  </div>
</header>
'''
    elements.append(header_html)

    # Generate promotional banners
    promo_messages = [
        f"✈️ Flash Sale! Save up to {rng.randint(20, 50)}% on select flights",
        f"🎉 Earn {rng.randint(2, 5)}x miles on all bookings this week",
        f"💳 New card members get {rng.randint(50000, 100000)} bonus miles",
        f"🏖️ Summer getaways from ${rng.randint(99, 299)} one-way",
    ]
    banner_html = f'''
<div class="promo-banner-container bg-gradient-to-r from-{rng.choice(['blue', 'indigo', 'purple'])}-{rng.randint(5,7)}00 to-{rng.choice(['blue', 'indigo', 'purple'])}-{rng.randint(6,8)}00" data-component="PromoBanner" data-banner-id="banner_{rng.randint(10000, 99999)}">
  <div class="promo-carousel" data-carousel-id="promo_{uuid.UUID(int=rng.getrandbits(128)).hex[:8]}">
'''
    for i, msg in enumerate(promo_messages):
        banner_html += f'''    <div class="promo-slide slide-{i}" data-slide-index="{i}" style="display: {'block' if i == 0 else 'none'}">
      <p class="promo-text text-white text-center py-{rng.randint(2, 4)}">{msg}</p>
      <a href="/deals/promo-{rng.randint(1000, 9999)}" class="promo-cta underline hover:no-underline" data-promo-code="SAVE{rng.randint(10, 50)}">Learn More</a>
    </div>
'''
    banner_html += '''  </div>
</div>
'''
    elements.append(banner_html)

    return '\n'.join(elements)


def _generate_tracking_pixels(rng: random.Random) -> str:
    """Generate advertising and tracking pixel data."""
    pixels = []

    # Google Analytics
    ga_id = f"G-{''.join(rng.choices(string.ascii_uppercase + string.digits, k=10))}"
    pixels.append(f'''
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id={ga_id}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{ga_id}', {{
    'page_title': 'Airline Booking - Results',
    'page_location': window.location.href,
    'page_path': window.location.pathname,
    'send_page_view': true,
    'cookie_domain': 'example-airline.com',
    'cookie_flags': 'SameSite=None;Secure',
    'custom_map': {{
      'dimension1': 'user_type',
      'dimension2': 'booking_flow',
      'dimension3': 'cabin_class',
      'metric1': 'search_results_count'
    }},
    'user_properties': {{
      'loyalty_tier': '{rng.choice(['bronze', 'silver', 'gold', 'platinum', 'guest'])}',
      'logged_in': {str(rng.random() > 0.5).lower()},
      'preferred_cabin': '{rng.choice(['economy', 'business', 'first'])}'
    }}
  }});
  gtag('event', 'page_view', {{
    'event_category': 'navigation',
    'event_label': 'api_response_rendered',
    'value': {rng.randint(1, 100)},
    'non_interaction': true
  }});
</script>
''')

    # Facebook Pixel
    fb_pixel_id = str(rng.randint(100000000000000, 999999999999999))
    pixels.append(f'''
<!-- Facebook Pixel Code -->
<script>
!function(f,b,e,v,n,t,s)
{{if(f.fbq)return;n=f.fbq=function(){{n.callMethod?
n.callMethod.apply(n,arguments):n.queue.push(arguments)}};
if(!f._fbq)f._fbq=n;n.push=n;n.loaded=!0;n.version='2.0';
n.queue=[];t=b.createElement(e);t.async=!0;
t.src=v;s=b.getElementsByTagName(e)[0];
s.parentNode.insertBefore(t,s)}}(window, document,'script',
'https://connect.facebook.net/en_US/fbevents.js');
fbq('init', '{fb_pixel_id}');
fbq('track', 'PageView');
fbq('track', 'Search', {{
  content_type: 'flight',
  content_ids: ['{uuid.UUID(int=rng.getrandbits(128)).hex[:12]}'],
  search_string: 'flight booking',
  currency: 'USD',
  value: {rng.randint(100, 2000)}
}});
</script>
<noscript><img height="1" width="1" style="display:none"
src="https://www.facebook.com/tr?id={fb_pixel_id}&ev=PageView&noscript=1"
/></noscript>
<!-- End Facebook Pixel Code -->
''')

    # Additional tracking pixels
    tracking_providers = ['doubleclick', 'criteo', 'taboola', 'outbrain', 'bing', 'linkedin', 'twitter', 'pinterest', 'snapchat', 'tiktok']
    for provider in rng.sample(tracking_providers, rng.randint(3, 6)):
        pixel_id = ''.join(rng.choices(string.ascii_uppercase + string.digits, k=rng.randint(8, 16)))
        pixels.append(f'''
<!-- {provider.title()} Tracking Pixel -->
<img src="https://tracking.{provider}.com/pixel?id={pixel_id}&event=pageview&t={rng.randint(1000000000, 9999999999)}&r={rng.random():.16f}" width="1" height="1" style="display:none" alt="" data-pixel-provider="{provider}" />
<script>
  window.__{provider}q = window.__{provider}q || [];
  window.__{provider}q.push(['track', 'PageView', {{
    'pixel_id': '{pixel_id}',
    'page_type': 'api_response',
    'session_id': 'sess_{uuid.UUID(int=rng.getrandbits(128)).hex[:16]}',
    'user_segment': '{rng.choice(['new', 'returning', 'loyal', 'dormant'])}',
    'experiment_id': 'exp_{rng.randint(1000, 9999)}',
    'variant': '{rng.choice(['control', 'treatment_a', 'treatment_b', 'treatment_c'])}'
  }}]);
</script>
''')

    return '\n'.join(pixels)


def _generate_cookie_consent_banner(rng: random.Random) -> str:
    """Generate cookie consent banner and GDPR compliance notices."""
    consent_id = uuid.UUID(int=rng.getrandbits(128)).hex[:16]
    return f'''
<!-- Cookie Consent Banner -->
<div id="cookie-consent-banner" class="cookie-banner fixed bottom-0 left-0 right-0 bg-white shadow-2xl z-[9999] p-{rng.randint(4, 8)} border-t border-gray-200" data-consent-version="{rng.randint(1, 5)}.{rng.randint(0, 9)}" data-consent-id="{consent_id}" style="display: none;">
  <div class="cookie-banner-content max-w-{rng.choice(['6xl', '7xl', 'full'])} mx-auto">
    <div class="flex flex-col lg:flex-row items-start lg:items-center gap-{rng.randint(4, 6)}">
      <div class="cookie-text flex-1">
        <h3 class="text-lg font-semibold mb-2">We value your privacy</h3>
        <p class="text-sm text-gray-600 mb-2">
          We use cookies and similar technologies to enhance your browsing experience, analyze site traffic,
          and personalize content. By clicking "Accept All", you consent to our use of cookies.
          You can manage your preferences by clicking "Cookie Settings".
        </p>
        <p class="text-xs text-gray-500">
          For more information, please read our
          <a href="/privacy-policy" class="text-blue-600 hover:underline">Privacy Policy</a> and
          <a href="/cookie-policy" class="text-blue-600 hover:underline">Cookie Policy</a>.
        </p>
      </div>
      <div class="cookie-actions flex flex-wrap gap-{rng.randint(2, 4)}">
        <button id="cookie-reject-all" class="px-{rng.randint(4, 6)} py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50 transition-colors" data-consent-action="reject">
          Reject All
        </button>
        <button id="cookie-settings" class="px-{rng.randint(4, 6)} py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50 transition-colors" data-consent-action="settings">
          Cookie Settings
        </button>
        <button id="cookie-accept-all" class="px-{rng.randint(4, 6)} py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors" data-consent-action="accept">
          Accept All
        </button>
      </div>
    </div>
  </div>
</div>

<!-- Cookie Preferences Modal -->
<div id="cookie-preferences-modal" class="cookie-modal hidden fixed inset-0 z-[10000] bg-black/50" data-modal-id="cookie-prefs-{consent_id}">
  <div class="modal-content bg-white max-w-2xl mx-auto mt-20 rounded-lg shadow-xl max-h-[80vh] overflow-y-auto">
    <div class="modal-header p-6 border-b">
      <h2 class="text-xl font-semibold">Cookie Preferences</h2>
      <button class="modal-close absolute top-4 right-4" aria-label="Close">×</button>
    </div>
    <div class="modal-body p-6">
      <div class="cookie-category mb-6">
        <div class="flex items-center justify-between mb-2">
          <h4 class="font-medium">Essential Cookies</h4>
          <span class="text-sm text-gray-500">Always Active</span>
        </div>
        <p class="text-sm text-gray-600">These cookies are necessary for the website to function and cannot be disabled. They are usually set in response to actions made by you such as setting your privacy preferences, logging in, or filling in forms.</p>
      </div>
      <div class="cookie-category mb-6">
        <div class="flex items-center justify-between mb-2">
          <h4 class="font-medium">Analytics Cookies</h4>
          <label class="toggle-switch">
            <input type="checkbox" id="analytics-cookies" checked data-category="analytics" />
            <span class="toggle-slider"></span>
          </label>
        </div>
        <p class="text-sm text-gray-600">These cookies allow us to count visits and traffic sources so we can measure and improve the performance of our site. They help us know which pages are the most and least popular.</p>
      </div>
      <div class="cookie-category mb-6">
        <div class="flex items-center justify-between mb-2">
          <h4 class="font-medium">Marketing Cookies</h4>
          <label class="toggle-switch">
            <input type="checkbox" id="marketing-cookies" data-category="marketing" />
            <span class="toggle-slider"></span>
          </label>
        </div>
        <p class="text-sm text-gray-600">These cookies may be set through our site by our advertising partners. They may be used to build a profile of your interests and show you relevant advertisements on other sites.</p>
      </div>
      <div class="cookie-category mb-6">
        <div class="flex items-center justify-between mb-2">
          <h4 class="font-medium">Functional Cookies</h4>
          <label class="toggle-switch">
            <input type="checkbox" id="functional-cookies" checked data-category="functional" />
            <span class="toggle-slider"></span>
          </label>
        </div>
        <p class="text-sm text-gray-600">These cookies enable the website to provide enhanced functionality and personalization. They may be set by us or by third-party providers whose services we have added to our pages.</p>
      </div>
    </div>
    <div class="modal-footer p-6 border-t bg-gray-50 flex justify-end gap-4">
      <button id="save-cookie-preferences" class="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">Save Preferences</button>
    </div>
  </div>
</div>

<script>
  window.__cookieConsent = {{
    version: '{rng.randint(1, 5)}.{rng.randint(0, 9)}',
    consentId: '{consent_id}',
    categories: {{
      essential: {{ enabled: true, required: true }},
      analytics: {{ enabled: {str(rng.random() > 0.3).lower()}, required: false }},
      marketing: {{ enabled: {str(rng.random() > 0.6).lower()}, required: false }},
      functional: {{ enabled: {str(rng.random() > 0.4).lower()}, required: false }}
    }},
    lastUpdated: '{rng.randint(2023, 2024)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z',
    gpcSignal: {str(rng.random() > 0.9).lower()},
    jurisdiction: '{rng.choice(['GDPR', 'CCPA', 'LGPD', 'POPIA'])}',
    tcfString: '{hashlib.sha256(str(rng.random()).encode()).hexdigest()[:40]}'
  }};
</script>
'''


def _generate_footer_content(rng: random.Random) -> str:
    """Generate typical website footer content."""
    return f'''
<!-- Site Footer -->
<footer class="site-footer bg-gray-{rng.choice([800, 900])} text-white mt-auto" data-component="GlobalFooter" data-footer-version="{rng.randint(1, 5)}.{rng.randint(0, 20)}">
  <div class="footer-main py-{rng.randint(12, 16)} px-{rng.randint(4, 8)}">
    <div class="max-w-7xl mx-auto">
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-{rng.randint(8, 12)}">

        <!-- Company Info Column -->
        <div class="footer-column company-info col-span-1 lg:col-span-2">
          <img src="/assets/logo-white-{hashlib.md5(str(rng.random()).encode()).hexdigest()[:6]}.svg" alt="Example Airlines" class="h-8 mb-4" />
          <p class="text-gray-400 text-sm mb-4">
            Example Airlines has been connecting travelers to destinations worldwide since {rng.randint(1950, 1990)}.
            With a fleet of over {rng.randint(200, 500)} aircraft serving {rng.randint(100, 300)}+ destinations,
            we're committed to making your journey comfortable and memorable.
          </p>
          <div class="social-links flex gap-4 mt-4">
            <a href="https://facebook.com/exampleairlines" class="social-link" aria-label="Facebook" data-social="facebook">
              <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>
            </a>
            <a href="https://twitter.com/exampleairlines" class="social-link" aria-label="Twitter" data-social="twitter">
              <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z"/></svg>
            </a>
            <a href="https://instagram.com/exampleairlines" class="social-link" aria-label="Instagram" data-social="instagram">
              <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0C8.74 0 8.333.015 7.053.072 5.775.132 4.905.333 4.14.63c-.789.306-1.459.717-2.126 1.384S.935 3.35.63 4.14C.333 4.905.131 5.775.072 7.053.012 8.333 0 8.74 0 12s.015 3.667.072 4.947c.06 1.277.261 2.148.558 2.913.306.788.717 1.459 1.384 2.126.667.666 1.336 1.079 2.126 1.384.766.296 1.636.499 2.913.558C8.333 23.988 8.74 24 12 24s3.667-.015 4.947-.072c1.277-.06 2.148-.262 2.913-.558.788-.306 1.459-.718 2.126-1.384.666-.667 1.079-1.335 1.384-2.126.296-.765.499-1.636.558-2.913.06-1.28.072-1.687.072-4.947s-.015-3.667-.072-4.947c-.06-1.277-.262-2.149-.558-2.913-.306-.789-.718-1.459-1.384-2.126C21.319 1.347 20.651.935 19.86.63c-.765-.297-1.636-.499-2.913-.558C15.667.012 15.26 0 12 0zm0 2.16c3.203 0 3.585.016 4.85.071 1.17.055 1.805.249 2.227.415.562.217.96.477 1.382.896.419.42.679.819.896 1.381.164.422.36 1.057.413 2.227.057 1.266.07 1.646.07 4.85s-.015 3.585-.074 4.85c-.061 1.17-.256 1.805-.421 2.227-.224.562-.479.96-.899 1.382-.419.419-.824.679-1.38.896-.42.164-1.065.36-2.235.413-1.274.057-1.649.07-4.859.07-3.211 0-3.586-.015-4.859-.074-1.171-.061-1.816-.256-2.236-.421-.569-.224-.96-.479-1.379-.899-.421-.419-.69-.824-.9-1.38-.165-.42-.359-1.065-.42-2.235-.045-1.26-.061-1.649-.061-4.844 0-3.196.016-3.586.061-4.861.061-1.17.255-1.814.42-2.234.21-.57.479-.96.9-1.381.419-.419.81-.689 1.379-.898.42-.166 1.051-.361 2.221-.421 1.275-.045 1.65-.06 4.859-.06l.045.03zm0 3.678c-3.405 0-6.162 2.76-6.162 6.162 0 3.405 2.76 6.162 6.162 6.162 3.405 0 6.162-2.76 6.162-6.162 0-3.405-2.757-6.162-6.162-6.162zM12 16c-2.21 0-4-1.79-4-4s1.79-4 4-4 4 1.79 4 4-1.79 4-4 4zm7.846-10.405c0 .795-.646 1.44-1.44 1.44-.795 0-1.44-.646-1.44-1.44 0-.794.646-1.439 1.44-1.439.793-.001 1.44.645 1.44 1.439z"/></svg>
            </a>
            <a href="https://linkedin.com/company/exampleairlines" class="social-link" aria-label="LinkedIn" data-social="linkedin">
              <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
            </a>
            <a href="https://youtube.com/exampleairlines" class="social-link" aria-label="YouTube" data-social="youtube">
              <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>
            </a>
          </div>
        </div>

        <!-- Quick Links Column -->
        <div class="footer-column">
          <h4 class="text-white font-semibold mb-4">Quick Links</h4>
          <ul class="space-y-2">
            <li><a href="/book" class="text-gray-400 hover:text-white text-sm transition-colors">Book a Flight</a></li>
            <li><a href="/check-in" class="text-gray-400 hover:text-white text-sm transition-colors">Online Check-in</a></li>
            <li><a href="/flight-status" class="text-gray-400 hover:text-white text-sm transition-colors">Flight Status</a></li>
            <li><a href="/manage-booking" class="text-gray-400 hover:text-white text-sm transition-colors">Manage Booking</a></li>
            <li><a href="/baggage" class="text-gray-400 hover:text-white text-sm transition-colors">Baggage Information</a></li>
            <li><a href="/seat-selection" class="text-gray-400 hover:text-white text-sm transition-colors">Seat Selection</a></li>
            <li><a href="/special-assistance" class="text-gray-400 hover:text-white text-sm transition-colors">Special Assistance</a></li>
          </ul>
        </div>

        <!-- Loyalty Program Column -->
        <div class="footer-column">
          <h4 class="text-white font-semibold mb-4">SkyRewards</h4>
          <ul class="space-y-2">
            <li><a href="/skyrewards" class="text-gray-400 hover:text-white text-sm transition-colors">Join SkyRewards</a></li>
            <li><a href="/skyrewards/earn" class="text-gray-400 hover:text-white text-sm transition-colors">Earn Miles</a></li>
            <li><a href="/skyrewards/redeem" class="text-gray-400 hover:text-white text-sm transition-colors">Redeem Miles</a></li>
            <li><a href="/skyrewards/tiers" class="text-gray-400 hover:text-white text-sm transition-colors">Membership Tiers</a></li>
            <li><a href="/skyrewards/partners" class="text-gray-400 hover:text-white text-sm transition-colors">Partners</a></li>
            <li><a href="/skyrewards/credit-card" class="text-gray-400 hover:text-white text-sm transition-colors">SkyRewards Credit Card</a></li>
          </ul>
        </div>

        <!-- Help & Support Column -->
        <div class="footer-column">
          <h4 class="text-white font-semibold mb-4">Help & Support</h4>
          <ul class="space-y-2">
            <li><a href="/help" class="text-gray-400 hover:text-white text-sm transition-colors">Help Center</a></li>
            <li><a href="/contact" class="text-gray-400 hover:text-white text-sm transition-colors">Contact Us</a></li>
            <li><a href="/faq" class="text-gray-400 hover:text-white text-sm transition-colors">FAQs</a></li>
            <li><a href="/refunds" class="text-gray-400 hover:text-white text-sm transition-colors">Refunds & Cancellations</a></li>
            <li><a href="/travel-advisories" class="text-gray-400 hover:text-white text-sm transition-colors">Travel Advisories</a></li>
            <li><a href="/accessibility" class="text-gray-400 hover:text-white text-sm transition-colors">Accessibility</a></li>
          </ul>
        </div>

      </div>
    </div>
  </div>

  <!-- Footer Bottom -->
  <div class="footer-bottom border-t border-gray-700 py-{rng.randint(4, 8)} px-{rng.randint(4, 8)}">
    <div class="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
      <p class="text-gray-500 text-sm">
        © {rng.randint(2020, 2024)} Example Airlines, Inc. All rights reserved.
      </p>
      <div class="footer-legal-links flex flex-wrap gap-4 text-sm">
        <a href="/privacy" class="text-gray-500 hover:text-white transition-colors">Privacy Policy</a>
        <a href="/terms" class="text-gray-500 hover:text-white transition-colors">Terms of Service</a>
        <a href="/cookies" class="text-gray-500 hover:text-white transition-colors">Cookie Policy</a>
        <a href="/do-not-sell" class="text-gray-500 hover:text-white transition-colors">Do Not Sell My Info</a>
        <a href="/sitemap" class="text-gray-500 hover:text-white transition-colors">Sitemap</a>
      </div>
    </div>
  </div>
</footer>
'''


def _generate_chat_widget(rng: random.Random) -> str:
    """Generate chat widget initialization code."""
    chat_id = uuid.UUID(int=rng.getrandbits(128)).hex[:16]
    return f'''
<!-- Live Chat Widget -->
<div id="chat-widget-container" class="fixed bottom-4 right-4 z-[9998]" data-widget-id="{chat_id}" data-widget-state="minimized">
  <div id="chat-widget-button" class="w-14 h-14 bg-blue-600 rounded-full shadow-lg cursor-pointer flex items-center justify-center hover:bg-blue-700 transition-colors" aria-label="Open chat">
    <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path>
    </svg>
    <span class="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full text-white text-xs flex items-center justify-center">{rng.randint(0, 3)}</span>
  </div>
  <div id="chat-widget-window" class="hidden absolute bottom-16 right-0 w-80 h-96 bg-white rounded-lg shadow-2xl flex flex-col overflow-hidden">
    <div class="chat-header bg-blue-600 text-white p-4 flex items-center justify-between">
      <div class="flex items-center gap-2">
        <div class="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center">
          <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z"/></svg>
        </div>
        <div>
          <p class="font-medium text-sm">Support Agent</p>
          <p class="text-xs text-blue-200">Online</p>
        </div>
      </div>
      <button id="chat-close" class="text-white/80 hover:text-white" aria-label="Close chat">×</button>
    </div>
    <div class="chat-messages flex-1 p-4 overflow-y-auto bg-gray-50">
      <div class="chat-message bot mb-4">
        <div class="bg-white p-3 rounded-lg shadow-sm max-w-[80%]">
          <p class="text-sm text-gray-700">Hi! How can I help you with your booking today?</p>
          <span class="text-xs text-gray-400 mt-1 block">{rng.randint(1, 12)}:{rng.randint(0, 59):02d} PM</span>
        </div>
      </div>
    </div>
    <div class="chat-input p-4 border-t bg-white">
      <div class="flex gap-2">
        <input type="text" placeholder="Type a message..." class="flex-1 px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
        <button class="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 transition-colors">Send</button>
      </div>
    </div>
  </div>
</div>

<script>
  window.__chatWidget = {{
    id: '{chat_id}',
    config: {{
      position: 'bottom-right',
      theme: 'blue',
      language: 'en',
      welcomeMessage: 'Hi! How can I help you with your booking today?',
      offlineMessage: 'Sorry, our agents are currently offline. Please leave a message.',
      inputPlaceholder: 'Type a message...',
      buttonText: 'Chat with us',
      headerTitle: 'Support Agent',
      preChatForm: {{
        enabled: {str(rng.random() > 0.5).lower()},
        fields: ['name', 'email', 'department']
      }},
      triggers: {{
        timeOnPage: {rng.randint(30, 120)},
        scrollPercentage: {rng.randint(50, 90)},
        exitIntent: {str(rng.random() > 0.5).lower()}
      }},
      businessHours: {{
        enabled: true,
        timezone: 'America/New_York',
        hours: {{
          monday: {{ start: '09:00', end: '17:00' }},
          tuesday: {{ start: '09:00', end: '17:00' }},
          wednesday: {{ start: '09:00', end: '17:00' }},
          thursday: {{ start: '09:00', end: '17:00' }},
          friday: {{ start: '09:00', end: '17:00' }},
          saturday: {{ start: '10:00', end: '14:00' }},
          sunday: null
        }}
      }},
      routing: {{
        department: '{rng.choice(['general', 'bookings', 'baggage', 'loyalty', 'complaints'])}',
        priority: '{rng.choice(['low', 'normal', 'high'])}',
        tags: ['web_visitor', 'api_response_page']
      }},
      analytics: {{
        trackPageViews: true,
        trackEvents: true,
        integrations: ['google_analytics', 'segment', 'mixpanel']
      }}
    }},
    session: {{
      id: 'chat_sess_{uuid.UUID(int=rng.getrandbits(128)).hex[:12]}',
      startedAt: {rng.randint(1715780000000, 1715790000000)},
      messages: {rng.randint(0, 5)},
      agentId: 'agent_{rng.randint(100, 999)}',
      queuePosition: {rng.randint(0, 5)},
      estimatedWaitTime: {rng.randint(0, 300)}
    }}
  }};
</script>
'''


def _generate_accessibility_tree(rng: random.Random) -> str:
    """Generate accessibility tree data like from an automated testing tool."""
    import json

    nodes = []
    node_id = 1

    def add_node(role, name, level=0, children_count=0):
        nonlocal node_id
        node = {
            "nodeId": node_id,
            "role": {"type": "role", "value": role},
            "name": {"type": "computedString", "value": name},
            "properties": [
                {"name": "focusable", "value": {"type": "booleanOrUndefined", "value": rng.random() > 0.5}},
                {"name": "focused", "value": {"type": "booleanOrUndefined", "value": False}},
                {"name": "level", "value": {"type": "integer", "value": level}} if level > 0 else None,
            ],
            "childIds": list(range(node_id + 1, node_id + 1 + children_count)) if children_count > 0 else [],
            "backendDOMNodeId": rng.randint(1, 10000),
        }
        node["properties"] = [p for p in node["properties"] if p is not None]
        nodes.append(node)
        node_id += 1
        return node

    # Build accessibility tree
    add_node("WebArea", "Airline Booking Results", children_count=5)
    add_node("banner", "Site header", children_count=2)
    add_node("navigation", "Main navigation", children_count=10)
    for item in ["Home", "Flights", "Hotels", "Cars", "Deals", "My Trips", "Check-in", "Status", "Help", "Account"]:
        add_node("link", item)
    add_node("img", "Example Airlines logo")
    add_node("main", "Main content", children_count=3)
    add_node("region", "Search results", children_count=2)
    add_node("heading", "Flight Search Results", level=1)
    add_node("list", "Available flights", children_count=5)
    for i in range(5):
        add_node("listitem", f"Flight option {i+1}", children_count=3)
        add_node("text", f"Departure: {rng.randint(6, 23):02d}:{rng.randint(0, 59):02d}")
        add_node("text", f"Price: ${rng.randint(100, 1500)}")
        add_node("button", "Select flight")
    add_node("complementary", "Sidebar", children_count=2)
    add_node("region", "Filters")
    add_node("region", "Sort options")
    add_node("contentinfo", "Site footer", children_count=4)
    for section in ["Quick Links", "Company", "Help", "Legal"]:
        add_node("navigation", section)

    return f'''
<!-- ACCESSIBILITY_TREE_SNAPSHOT_BEGIN -->
<!-- Generated by automated accessibility testing tool -->
<!-- Snapshot timestamp: 2024-05-15T{rng.randint(10, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z -->
<!-- WCAG Level: AA -->
<!-- Tool version: axe-core@{rng.randint(4, 5)}.{rng.randint(0, 9)}.{rng.randint(0, 9)} -->
<script type="application/json" id="accessibility-tree-data">
{{
  "version": "1.0",
  "agent": "Chrome/{rng.randint(100, 125)}.0.{rng.randint(0, 9999)}.{rng.randint(0, 999)}",
  "url": "https://www.example-airline.com/api/response",
  "timestamp": {rng.randint(1715780000000, 1715790000000)},
  "tree": {json.dumps(nodes, indent=2)},
  "issues": [
    {{
      "id": "color-contrast",
      "impact": "serious",
      "description": "Elements must have sufficient color contrast",
      "nodes": {rng.randint(0, 5)}
    }},
    {{
      "id": "image-alt",
      "impact": "critical",
      "description": "Images must have alternate text",
      "nodes": {rng.randint(0, 2)}
    }},
    {{
      "id": "link-name",
      "impact": "serious",
      "description": "Links must have discernible text",
      "nodes": {rng.randint(0, 3)}
    }},
    {{
      "id": "button-name",
      "impact": "critical",
      "description": "Buttons must have discernible text",
      "nodes": {rng.randint(0, 1)}
    }}
  ],
  "stats": {{
    "totalNodes": {len(nodes)},
    "interactiveElements": {rng.randint(20, 50)},
    "landmarkRegions": {rng.randint(5, 10)},
    "headings": {rng.randint(3, 15)},
    "links": {rng.randint(30, 80)},
    "buttons": {rng.randint(10, 30)},
    "forms": {rng.randint(1, 5)},
    "images": {rng.randint(5, 20)}
  }}
}}
</script>
<!-- ACCESSIBILITY_TREE_SNAPSHOT_END -->
'''


def _generate_performance_metrics(rng: random.Random) -> str:
    """Generate detailed performance timing metrics."""
    return f'''
<!-- PERFORMANCE_TIMING_DATA_BEGIN -->
<script type="application/json" id="performance-timing">
{{
  "navigationTiming": {{
    "navigationStart": {rng.randint(1715780000000, 1715781000000)},
    "unloadEventStart": 0,
    "unloadEventEnd": 0,
    "redirectStart": 0,
    "redirectEnd": 0,
    "fetchStart": {rng.randint(1, 50)},
    "domainLookupStart": {rng.randint(50, 100)},
    "domainLookupEnd": {rng.randint(100, 150)},
    "connectStart": {rng.randint(150, 200)},
    "connectEnd": {rng.randint(200, 300)},
    "secureConnectionStart": {rng.randint(180, 250)},
    "requestStart": {rng.randint(300, 400)},
    "responseStart": {rng.randint(400, 600)},
    "responseEnd": {rng.randint(600, 800)},
    "domLoading": {rng.randint(800, 1000)},
    "domInteractive": {rng.randint(1000, 1500)},
    "domContentLoadedEventStart": {rng.randint(1500, 2000)},
    "domContentLoadedEventEnd": {rng.randint(2000, 2200)},
    "domComplete": {rng.randint(2200, 3000)},
    "loadEventStart": {rng.randint(3000, 3500)},
    "loadEventEnd": {rng.randint(3500, 4000)}
  }},
  "resourceTiming": [
    {{
      "name": "https://cdn.example-airline.com/js/main.{hashlib.md5(str(rng.random()).encode()).hexdigest()[:8]}.js",
      "entryType": "resource",
      "startTime": {rng.randint(100, 300)},
      "duration": {rng.randint(50, 200)},
      "initiatorType": "script",
      "transferSize": {rng.randint(50000, 200000)},
      "encodedBodySize": {rng.randint(40000, 180000)},
      "decodedBodySize": {rng.randint(100000, 500000)}
    }},
    {{
      "name": "https://cdn.example-airline.com/css/styles.{hashlib.md5(str(rng.random()).encode()).hexdigest()[:8]}.css",
      "entryType": "resource",
      "startTime": {rng.randint(100, 200)},
      "duration": {rng.randint(30, 100)},
      "initiatorType": "link",
      "transferSize": {rng.randint(10000, 50000)},
      "encodedBodySize": {rng.randint(8000, 45000)},
      "decodedBodySize": {rng.randint(20000, 100000)}
    }},
    {{
      "name": "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
      "entryType": "resource",
      "startTime": {rng.randint(150, 250)},
      "duration": {rng.randint(100, 300)},
      "initiatorType": "link",
      "transferSize": {rng.randint(1000, 5000)},
      "encodedBodySize": {rng.randint(800, 4500)},
      "decodedBodySize": {rng.randint(1500, 8000)}
    }},
    {{
      "name": "https://www.googletagmanager.com/gtag/js?id=G-{''.join(rng.choices(string.ascii_uppercase + string.digits, k=10))}",
      "entryType": "resource",
      "startTime": {rng.randint(200, 400)},
      "duration": {rng.randint(100, 400)},
      "initiatorType": "script",
      "transferSize": {rng.randint(30000, 80000)},
      "encodedBodySize": {rng.randint(25000, 70000)},
      "decodedBodySize": {rng.randint(80000, 200000)}
    }}
  ],
  "paintTiming": {{
    "first-paint": {rng.randint(500, 1000)},
    "first-contentful-paint": {rng.randint(600, 1200)}
  }},
  "largestContentfulPaint": {{
    "renderTime": {rng.randint(1000, 2500)},
    "loadTime": {rng.randint(800, 2000)},
    "size": {rng.randint(50000, 200000)},
    "element": "IMG"
  }},
  "layoutShift": {{
    "value": {rng.random() * 0.15:.4f},
    "hadRecentInput": false,
    "sources": [
      {{
        "node": "DIV.flight-results",
        "previousRect": {{"x": 0, "y": {rng.randint(100, 300)}, "width": {rng.randint(800, 1200)}, "height": {rng.randint(200, 400)}}},
        "currentRect": {{"x": 0, "y": {rng.randint(150, 350)}, "width": {rng.randint(800, 1200)}, "height": {rng.randint(200, 400)}}}
      }}
    ]
  }},
  "firstInputDelay": {{
    "value": {rng.randint(10, 100)},
    "startTime": {rng.randint(2000, 4000)},
    "processingStart": {rng.randint(2010, 4020)},
    "processingEnd": {rng.randint(2015, 4030)},
    "target": "BUTTON"
  }},
  "longTasks": [
    {{
      "name": "self",
      "entryType": "longtask",
      "startTime": {rng.randint(500, 1500)},
      "duration": {rng.randint(60, 200)},
      "attribution": [{{
        "name": "script",
        "entryType": "taskattribution",
        "containerType": "iframe",
        "containerSrc": "https://www.googletagmanager.com/gtag/js"
      }}]
    }}
  ],
  "memory": {{
    "usedJSHeapSize": {rng.randint(10000000, 50000000)},
    "totalJSHeapSize": {rng.randint(50000000, 100000000)},
    "jsHeapSizeLimit": {rng.randint(2000000000, 4000000000)}
  }},
  "webVitals": {{
    "LCP": {rng.randint(1000, 2500)},
    "FID": {rng.randint(10, 100)},
    "CLS": {rng.random() * 0.15:.4f},
    "TTFB": {rng.randint(200, 600)},
    "FCP": {rng.randint(600, 1200)},
    "INP": {rng.randint(50, 300)},
    "rating": "{rng.choice(['good', 'needs-improvement', 'poor'])}"
  }}
}}
</script>
<!-- PERFORMANCE_TIMING_DATA_END -->
'''


def _generate_ab_test_data(rng: random.Random) -> str:
    """Generate A/B testing and experimentation data."""
    experiments = []
    experiment_names = [
        "new_search_flow", "pricing_display_v2", "checkout_optimization",
        "loyalty_banner_position", "flight_card_design", "sort_default_order",
        "filter_ui_experiment", "mobile_nav_update", "booking_summary_layout",
        "payment_options_order", "seat_selection_ui", "baggage_upsell_timing"
    ]

    for name in rng.sample(experiment_names, rng.randint(3, 6)):
        exp = {
            "experimentId": f"exp_{hashlib.md5(name.encode()).hexdigest()[:12]}",
            "experimentName": name,
            "variant": rng.choice(["control", "treatment_a", "treatment_b", "treatment_c"]),
            "variantId": f"var_{rng.randint(1, 4)}",
            "enrolled": True,
            "eligibility": {
                "userSegment": rng.choice(["all", "new_users", "returning", "loyalty_members"]),
                "trafficPercentage": rng.randint(10, 100),
                "geoTargeting": rng.choice(["US", "NA", "GLOBAL", None]),
                "deviceType": rng.choice(["all", "desktop", "mobile", "tablet"])
            },
            "metadata": {
                "startDate": f"2024-0{rng.randint(1, 5)}-{rng.randint(1, 28):02d}",
                "endDate": f"2024-0{rng.randint(6, 9)}-{rng.randint(1, 28):02d}",
                "owner": f"team_{rng.choice(['growth', 'product', 'ux', 'conversion'])}",
                "hypothesis": f"Changing {name.replace('_', ' ')} will improve conversion by {rng.randint(5, 25)}%"
            }
        }
        experiments.append(exp)

    import json
    return f'''
<!-- A/B_TESTING_DATA_BEGIN -->
<script type="application/json" id="ab-testing-data">
{{
  "platform": "optimizely",
  "platformVersion": "{rng.randint(1, 5)}.{rng.randint(0, 99)}.{rng.randint(0, 999)}",
  "visitorId": "vis_{uuid.UUID(int=rng.getrandbits(128)).hex[:16]}",
  "sessionId": "sess_{uuid.UUID(int=rng.getrandbits(128)).hex[:12]}",
  "bucketingId": "bucket_{hashlib.sha256(str(rng.random()).encode()).hexdigest()[:16]}",
  "activeExperiments": {json.dumps(experiments, indent=2)},
  "featureFlags": {{
    "new_booking_engine": {str(rng.random() > 0.5).lower()},
    "enhanced_search": {str(rng.random() > 0.3).lower()},
    "loyalty_integration_v2": {str(rng.random() > 0.6).lower()},
    "real_time_pricing": {str(rng.random() > 0.4).lower()},
    "ai_recommendations": {str(rng.random() > 0.7).lower()},
    "dark_mode": {str(rng.random() > 0.8).lower()},
    "chat_support": {str(rng.random() > 0.5).lower()},
    "price_alerts": {str(rng.random() > 0.6).lower()}
  }},
  "segmentation": {{
    "userType": "{rng.choice(['new', 'returning', 'loyal', 'dormant'])}",
    "loyaltyTier": "{rng.choice(['none', 'bronze', 'silver', 'gold', 'platinum'])}",
    "lifetimeValue": "{rng.choice(['low', 'medium', 'high', 'vip'])}",
    "bookingFrequency": "{rng.choice(['rare', 'occasional', 'frequent', 'very_frequent'])}",
    "preferredCabin": "{rng.choice(['economy', 'premium_economy', 'business', 'first', 'mixed'])}",
    "deviceCategory": "{rng.choice(['desktop', 'mobile', 'tablet'])}",
    "browser": "{rng.choice(['chrome', 'firefox', 'safari', 'edge'])}",
    "os": "{rng.choice(['windows', 'macos', 'ios', 'android', 'linux'])}"
  }},
  "decisionLog": [
    {{
      "timestamp": {rng.randint(1715780000, 1715790000)},
      "experimentId": "exp_{hashlib.md5(str(rng.random()).encode()).hexdigest()[:12]}",
      "decision": "enrolled",
      "reason": "user_eligible_traffic_allocated"
    }},
    {{
      "timestamp": {rng.randint(1715780000, 1715790000)},
      "experimentId": "exp_{hashlib.md5(str(rng.random()).encode()).hexdigest()[:12]}",
      "decision": "excluded",
      "reason": "traffic_not_allocated"
    }}
  ]
}}
</script>
<!-- A/B_TESTING_DATA_END -->
'''


def _generate_session_replay_data(rng: random.Random) -> str:
    """Generate session replay recording metadata."""
    session_id = uuid.UUID(int=rng.getrandbits(128)).hex[:16]
    return f'''
<!-- SESSION_REPLAY_METADATA_BEGIN -->
<!-- Session recording powered by FullStory/LogRocket/Hotjar simulation -->
<script type="application/json" id="session-replay-meta">
{{
  "provider": "{rng.choice(['fullstory', 'logrocket', 'hotjar', 'clarity', 'heap'])}",
  "sessionId": "{session_id}",
  "recordingId": "rec_{uuid.UUID(int=rng.getrandbits(128)).hex[:20]}",
  "userId": "{rng.choice(['anonymous', f'user_{rng.randint(100000, 999999)}'])}",
  "sessionStartTime": {rng.randint(1715780000000, 1715781000000)},
  "currentTime": {rng.randint(1715781000000, 1715790000000)},
  "pageViews": {rng.randint(1, 10)},
  "events": {rng.randint(50, 500)},
  "clicks": {rng.randint(10, 100)},
  "scrollDepth": {rng.randint(20, 100)},
  "rageClicks": {rng.randint(0, 5)},
  "deadClicks": {rng.randint(0, 10)},
  "errorCount": {rng.randint(0, 3)},
  "consoleErrors": {rng.randint(0, 5)},
  "networkErrors": {rng.randint(0, 2)},
  "frustrationScore": {rng.randint(0, 100)},
  "engagementScore": {rng.randint(30, 100)},
  "recording": {{
    "hasVideo": true,
    "hasAudio": false,
    "resolution": "{rng.choice(['1920x1080', '1440x900', '1366x768', '1280x720'])}",
    "fps": {rng.choice([10, 15, 30])},
    "codec": "h264",
    "estimatedSize": "{rng.randint(1, 10)}.{rng.randint(0, 9)}MB"
  }},
  "privacy": {{
    "maskInputs": true,
    "maskText": {str(rng.random() > 0.5).lower()},
    "blockClass": "fs-block",
    "ignoreClass": "fs-ignore",
    "maskAllText": false,
    "maskAllInputs": true
  }},
  "device": {{
    "type": "{rng.choice(['desktop', 'mobile', 'tablet'])}",
    "os": "{rng.choice(['Windows 10', 'macOS 14', 'iOS 17', 'Android 14'])}",
    "browser": "{rng.choice(['Chrome 124', 'Firefox 125', 'Safari 17', 'Edge 124'])}",
    "screenResolution": "{rng.choice(['1920x1080', '2560x1440', '1440x900', '390x844'])}",
    "viewportSize": "{rng.choice(['1920x937', '1440x821', '1366x625', '390x664'])}",
    "devicePixelRatio": {rng.choice([1, 2, 3])}
  }},
  "location": {{
    "country": "{rng.choice(['US', 'CA', 'GB', 'DE', 'FR', 'AU'])}",
    "region": "{rng.choice(['CA', 'NY', 'TX', 'WA', 'FL'])}",
    "city": "{rng.choice(['San Francisco', 'New York', 'Los Angeles', 'Chicago', 'Seattle'])}",
    "timezone": "{rng.choice(['America/Los_Angeles', 'America/New_York', 'America/Chicago', 'Europe/London'])}"
  }},
  "utm": {{
    "source": "{rng.choice(['google', 'facebook', 'email', 'direct', None])}",
    "medium": "{rng.choice(['cpc', 'organic', 'social', 'email', None])}",
    "campaign": "{rng.choice(['spring_sale', 'brand', 'retargeting', None])}",
    "term": "{rng.choice(['cheap flights', 'airline tickets', None])}",
    "content": "{rng.choice(['banner_a', 'text_ad', None])}"
  }},
  "replayUrl": "https://app.{rng.choice(['fullstory', 'logrocket', 'hotjar'])}.com/replay/{session_id}"
}}
</script>
<!-- SESSION_REPLAY_METADATA_END -->
'''


def _generate_error_tracking_noise(rng: random.Random) -> str:
    """Generate error tracking and monitoring service data."""
    return f'''
<!-- ERROR_TRACKING_DATA_BEGIN -->
<script type="application/json" id="error-tracking-data">
{{
  "service": "{rng.choice(['sentry', 'bugsnag', 'rollbar', 'datadog', 'newrelic'])}",
  "environment": "{rng.choice(['production', 'staging', 'development'])}",
  "release": "v{rng.randint(1, 5)}.{rng.randint(0, 99)}.{rng.randint(0, 999)}",
  "dsn": "https://{hashlib.md5(str(rng.random()).encode()).hexdigest()[:32]}@sentry.io/{rng.randint(100000, 999999)}",
  "tracesSampleRate": {rng.choice([0.1, 0.2, 0.5, 1.0])},
  "profilesSampleRate": {rng.choice([0.1, 0.2, 0.5])},
  "replaysSampleRate": {rng.choice([0.1, 0.2, 0.5])},
  "currentTransaction": {{
    "traceId": "{uuid.UUID(int=rng.getrandbits(128)).hex}",
    "spanId": "{uuid.UUID(int=rng.getrandbits(128)).hex[:16]}",
    "parentSpanId": "{uuid.UUID(int=rng.getrandbits(128)).hex[:16]}",
    "name": "api.response.render",
    "op": "http.client",
    "status": "ok",
    "startTimestamp": {rng.randint(1715780000, 1715790000)}.{rng.randint(100, 999)},
    "timestamp": {rng.randint(1715780000, 1715790000)}.{rng.randint(100, 999)}
  }},
  "breadcrumbs": [
    {{"type": "navigation", "category": "navigation", "data": {{"from": "/search", "to": "/results"}}, "timestamp": {rng.randint(1715780000, 1715790000)}}},
    {{"type": "http", "category": "xhr", "data": {{"method": "GET", "url": "/api/flights", "status_code": 200}}, "timestamp": {rng.randint(1715780000, 1715790000)}}},
    {{"type": "ui.click", "category": "ui.click", "message": "button.search-submit", "timestamp": {rng.randint(1715780000, 1715790000)}}},
    {{"type": "console", "category": "console", "level": "log", "message": "Search completed", "timestamp": {rng.randint(1715780000, 1715790000)}}}
  ],
  "context": {{
    "browser": {{
      "name": "{rng.choice(['Chrome', 'Firefox', 'Safari', 'Edge'])}",
      "version": "{rng.randint(100, 125)}.0.{rng.randint(0, 9999)}.{rng.randint(0, 999)}"
    }},
    "os": {{
      "name": "{rng.choice(['Windows', 'macOS', 'Linux', 'iOS', 'Android'])}",
      "version": "{rng.randint(10, 14)}.{rng.randint(0, 5)}"
    }},
    "device": {{
      "family": "{rng.choice(['Desktop', 'iPhone', 'Android', 'iPad'])}",
      "model": "{rng.choice(['MacBook Pro', 'iPhone 15', 'Pixel 8', 'iPad Pro', 'ThinkPad'])}"
    }}
  }},
  "user": {{
    "id": "{rng.choice([f'user_{rng.randint(100000, 999999)}', None])}",
    "email": null,
    "ip_address": "{{{{auto}}}}",
    "segment": "{rng.choice(['free', 'premium', 'enterprise'])}"
  }},
  "tags": {{
    "page": "api_response",
    "feature": "booking_flow",
    "experiment": "exp_{hashlib.md5(str(rng.random()).encode()).hexdigest()[:8]}"
  }}
}}
</script>
<!-- ERROR_TRACKING_DATA_END -->
'''


def _add_noise_to_result(result: Any, operation_name: str, seed_key: str) -> str:
    """Wrap a tool result with realistic noise from web scraping or raw traces.

    Args:
        result: The actual tool result data
        operation_name: Name of the operation for metadata
        seed_key: A string derived from tool call ARGUMENTS (not results) to ensure
                  deterministic noise across simulation and evaluation replay
    """
    import json

    # Convert pydantic models or other objects to dict
    if hasattr(result, 'model_dump'):
        result_data = result.model_dump()
    elif isinstance(result, list):
        result_data = [item.model_dump() if hasattr(item, 'model_dump') else item for item in result]
    else:
        result_data = result

    # Create a local Random instance seeded based on ARGUMENTS (seed_key), not results
    # This ensures deterministic noise for same tool calls (critical for evaluation replay)
    seed_str = operation_name + ":" + seed_key
    seed_value = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed_value)

    # Build the noisy output
    noise_output = []

    # Add HTML noise at the beginning
    noise_output.append(_generate_html_noise(rng))

    # Add request metadata
    noise_output.append(_generate_request_metadata(rng))

    # Start JSON trace wrapper
    noise_output.append(_generate_json_trace_noise(rng))

    # Add legacy fields
    noise_output.append(_generate_legacy_field_noise(rng))

    # Add verbose timestamps
    noise_output.append(_generate_verbose_timestamps(rng))

    # Add internal IDs
    noise_output.append(_generate_internal_ids(rng))

    # Add operation-specific metadata
    noise_output.append(f'''
  "_operation": {{
    "name": "{operation_name}",
    "type": "{rng.choice(['query', 'mutation', 'read', 'write'])}",
    "idempotency_key": "idem_{uuid.UUID(int=rng.getrandbits(128)).hex[:16]}",
    "retry_count": 0,
    "max_retries": 3,
    "timeout_ms": {rng.randint(5000, 30000)},
    "circuit_breaker_status": "closed"
  }},
''')

    # Add the actual data
    noise_output.append(f'''
  "data": {json.dumps(result_data, indent=4, default=str)},
''')

    # Add status and response metadata
    noise_output.append(f'''
  "_response": {{
    "status": "success",
    "status_code": 200,
    "message": "Operation completed successfully",
    "error": null,
    "error_code": null,
    "error_details": null,
    "warnings": [],
    "info": [
      "Response generated by airline-api-v2.{rng.randint(10,99)}.{rng.randint(0,999)}",
      "Processed by handler: {operation_name}Handler",
      "Backend latency within acceptable range"
    ]
  }},
''')

    # Close JSON trace
    noise_output.append(_generate_json_trace_noise_end(rng))

    # Add more HTML noise at the end
    noise_output.append(f'''
<!-- END_API_RESPONSE -->
<div class="clearfix"></div>
<!-- analytics: page_view_id={uuid.UUID(int=rng.getrandbits(128)).hex[:16]} session_id={uuid.UUID(int=rng.getrandbits(128)).hex[:24]} -->
<!-- performance: dns={rng.randint(1,50)}ms tcp={rng.randint(10,100)}ms ttfb={rng.randint(50,300)}ms -->
<script>window.__INITIAL_STATE__=window.__INITIAL_STATE__||{{}};window.__INITIAL_STATE__.apiResponse=true;</script>
<!-- Served by: web-{rng.randint(1,50):03d}.{rng.choice(['us-west-2', 'us-east-1'])}.example-airline.com -->
''')

    # ============================================================================
    # MASSIVE NOISE INJECTION - Raw scraped web content simulation
    # These add thousands of extra tokens to simulate real-world web scraping
    # ============================================================================

    # Add raw click/interaction trace events (like from a session recorder)
    noise_output.append(_generate_click_trace_events(rng))

    # Add DOM snapshot (header, navigation, promotional banners)
    noise_output.append(_generate_dom_snapshot(rng))

    # Add tracking pixels (Google Analytics, Facebook, etc.)
    noise_output.append(_generate_tracking_pixels(rng))

    # Add cookie consent banner and GDPR notices
    noise_output.append(_generate_cookie_consent_banner(rng))

    # Add website footer content
    noise_output.append(_generate_footer_content(rng))

    # Add chat widget initialization
    noise_output.append(_generate_chat_widget(rng))

    # Add accessibility tree snapshot
    noise_output.append(_generate_accessibility_tree(rng))

    # Add detailed performance metrics
    noise_output.append(_generate_performance_metrics(rng))

    # Add A/B testing and experimentation data
    noise_output.append(_generate_ab_test_data(rng))

    # Add session replay metadata
    noise_output.append(_generate_session_replay_data(rng))

    # Add error tracking/monitoring data
    noise_output.append(_generate_error_tracking_noise(rng))

    return '\n'.join(noise_output)


class AirlineTools(ToolKitBase):  # Tools
    """All the tools for the airline domain."""

    db: FlightDB

    def __init__(self, db: FlightDB) -> None:
        super().__init__(db)

    def _get_user(self, user_id: str) -> User:
        """Get user from database."""
        if user_id not in self.db.users:
            raise ValueError(f"User {user_id} not found")
        return self.db.users[user_id]

    def _get_reservation(self, reservation_id: str) -> Reservation:
        """Get reservation from database."""
        if reservation_id not in self.db.reservations:
            raise ValueError(f"Reservation {reservation_id} not found")
        return self.db.reservations[reservation_id]

    def _get_flight(self, flight_number: str) -> Flight:
        """Get flight from database."""
        if flight_number not in self.db.flights:
            raise ValueError(f"Flight {flight_number} not found")
        return self.db.flights[flight_number]

    def _get_flight_instance(self, flight_number: str, date: str) -> FlightDateStatus:
        """Get flight instance from database."""
        flight = self._get_flight(flight_number)
        if date not in flight.dates:
            raise ValueError(f"Flight {flight_number} not found on date {date}")
        return flight.dates[date]

    def _get_flights_from_flight_infos(
        self, flight_infos: List[FlightInfo]
    ) -> list[FlightDateStatus]:
        """Get the flight from the reservation."""
        flights = []
        for flight_info in flight_infos:
            flights.append(
                self._get_flight_instance(flight_info.flight_number, flight_info.date)
            )
        return flights

    def _get_new_reservation_id(self) -> str:
        """Get a new reservation id.
        Assume each task makes at most 3 reservations

        Returns:
            A new reservation id.

        Raises:
            ValueError: If too many reservations are made.
        """
        for reservation_id in ["HATHAT", "HATHAU", "HATHAV"]:
            if reservation_id not in self.db.reservations:
                return reservation_id
        raise ValueError("Too many reservations")

    def _get_new_payment_id(self) -> str:
        """Get a new payment id.
        Assume each task makes at most 3 payments

        Returns:
            A new payment id.
        """
        return [3221322, 3221323, 3221324]

    def _get_datetime(self) -> str:
        """Get the current datetime."""
        return "2024-05-15T15:00:00"

    def _search_direct_flight(
        self,
        date: str,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        leave_after: Optional[str] = None,
    ) -> list[DirectFlight]:
        """Search for direct flights

        Args:
            date: The date of the flight in the format 'YYYY-MM-DD', such as '2024-01-01'.
            origin: The origin city airport in three letters, such as 'JFK'.
            destination: The destination city airport in three letters, such as 'LAX'.
            leave_after: The time to leave after the flight, such as '15:00:00'.
        """
        results = []
        for flight in self.db.flights.values():
            check = (
                (origin is None or flight.origin == origin)
                and (destination is None or flight.destination == destination)
                and (date in flight.dates)
                and (flight.dates[date].status == "available")
                and (
                    leave_after is None
                    or flight.scheduled_departure_time_est >= leave_after
                )
            )
            if check:
                direct_flight = DirectFlight(
                    flight_number=flight.flight_number,
                    origin=flight.origin,
                    destination=flight.destination,
                    status="available",
                    scheduled_departure_time_est=flight.scheduled_departure_time_est,
                    scheduled_arrival_time_est=flight.scheduled_arrival_time_est,
                    available_seats=flight.dates[date].available_seats,
                    prices=flight.dates[date].prices,
                )
                results.append(direct_flight)
        return results

    def _payment_for_update(
        self, user: User, payment_id: str, total_price: int
    ) -> Optional[Payment]:
        """
        Process payment for update reservation

        Args:
            user: The user to process payment for.
            payment_id: The payment id to process.
            total_price: The total price to process.
            reservation: The reservation to process payment for.

        Raises:
            ValueError: If the payment method is not found.
            ValueError: If the certificate is used to update reservation.
            ValueError: If the gift card balance is not enough.
        """
        # Check payment
        if payment_id not in user.payment_methods:
            raise ValueError("Payment method not found")
        payment_method = user.payment_methods[payment_id]
        if payment_method.source == "certificate":
            raise ValueError("Certificate cannot be used to update reservation")
        elif (
            payment_method.source == "gift_card" and payment_method.amount < total_price
        ):
            raise ValueError("Gift card balance is not enough")

        # Deduct payment
        if payment_method.source == "gift_card":
            payment_method.amount -= total_price

        payment = None
        # Create payment if total price is not 0
        if total_price != 0:
            payment = Payment(
                payment_id=payment_id,
                amount=total_price,
            )
        return payment

    @is_tool(ToolType.WRITE)
    def book_reservation(
        self,
        user_id: str,
        origin: str,
        destination: str,
        flight_type: FlightType,
        cabin: CabinClass,
        flights: List[FlightInfo | dict],
        passengers: List[Passenger | dict],
        payment_methods: List[Payment | dict],
        total_baggages: int,
        nonfree_baggages: int,
        insurance: Insurance,
    ) -> str:
        """
        Book a reservation.

        Args:
            user_id: The ID of the user to book the reservation such as 'sara_doe_496'`.
            origin: The IATA code for the origin city such as 'SFO'.
            destination: The IATA code for the destination city such as 'JFK'.
            flight_type: The type of flight such as 'one_way' or 'round_trip'.
            cabin: The cabin class such as 'basic_economy', 'economy', or 'business'.
            flights: An array of objects containing details about each piece of flight.
            passengers: An array of objects containing details about each passenger.
            payment_methods: An array of objects containing details about each payment method.
            total_baggages: The total number of baggage items to book the reservation.
            nonfree_baggages: The number of non-free baggage items to book the reservation.
            insurance: Whether the reservation has insurance.
        """
        if all(isinstance(flight, dict) for flight in flights):
            flights = [FlightInfo(**flight) for flight in flights]
        if all(isinstance(passenger, dict) for passenger in passengers):
            passengers = [Passenger(**passenger) for passenger in passengers]
        if all(isinstance(payment_method, dict) for payment_method in payment_methods):
            payment_methods = [
                Payment(**payment_method) for payment_method in payment_methods
            ]
        user = self._get_user(user_id)
        reservation_id = self._get_new_reservation_id()

        reservation = Reservation(
            reservation_id=reservation_id,
            user_id=user_id,
            origin=origin,
            destination=destination,
            flight_type=flight_type,
            cabin=cabin,
            flights=[],
            passengers=deepcopy(passengers),
            payment_history=deepcopy(payment_methods),
            created_at=self._get_datetime(),
            total_baggages=total_baggages,
            nonfree_baggages=nonfree_baggages,
            insurance=insurance,
        )

        # Update flights and calculate price
        total_price = 0
        all_flights_date_data: list[FlightDateStatusAvailable] = []

        for flight_info in flights:
            flight_number = flight_info.flight_number
            flight = self._get_flight(flight_number)
            flight_date_data = self._get_flight_instance(
                flight_number=flight_number, date=flight_info.date
            )
            # Checking flight availability
            if not isinstance(flight_date_data, FlightDateStatusAvailable):
                raise ValueError(
                    f"Flight {flight_number} not available on date {flight_info.date}"
                )
            # Checking seat availability
            if flight_date_data.available_seats[cabin] < len(passengers):
                raise ValueError(f"Not enough seats on flight {flight_number}")
            # Calculate price
            price = flight_date_data.prices[cabin]
            # Update reservation
            reservation.flights.append(
                ReservationFlight(
                    origin=flight.origin,
                    destination=flight.destination,
                    flight_number=flight_number,
                    date=flight_info.date,
                    price=price,
                )
            )
            all_flights_date_data.append(flight_date_data)
            total_price += price * len(passengers)

        # Add insurance fee
        if insurance == "yes":
            total_price += 30 * len(passengers)

        # Add baggage fee
        total_price += 50 * nonfree_baggages

        for payment_method in payment_methods:
            payment_id = payment_method.payment_id
            amount = payment_method.amount
            if payment_id not in user.payment_methods:
                raise ValueError(f"Payment method {payment_id} not found")

            user_payment_method = user.payment_methods[payment_id]
            if user_payment_method.source in {"gift_card", "certificate"}:
                if user_payment_method.amount < amount:
                    raise ValueError(
                        f"Not enough balance in payment method {payment_id}"
                    )

        total_payment = sum(payment.amount for payment in payment_methods)
        if total_payment != total_price:
            raise ValueError(
                f"Payment amount does not add up, total price is {total_price}, but paid {total_payment}"
            )

        # if checks pass, deduct payment
        for payment_method in payment_methods:
            payment_id = payment_method.payment_id
            amount = payment_method.amount
            user_payment_method = user.payment_methods[payment_id]
            if user_payment_method.source == "gift_card":
                user_payment_method.amount -= amount
            elif user_payment_method.source == "certificate":
                user.payment_methods.pop(payment_id)

        # Update DB
        for flight_date_data in all_flights_date_data:
            flight_date_data.available_seats[cabin] -= len(passengers)
        self.db.reservations[reservation_id] = reservation
        self.db.users[user_id].reservations.append(reservation_id)
        return _add_noise_to_result(reservation, "book_reservation", f"{user_id}:{origin}:{destination}")

    @is_tool(ToolType.GENERIC)
    def calculate(self, expression: str) -> str:
        """
        Calculate the result of a mathematical expression.

        Args:
            expression: The mathematical expression to calculate, such as '2 + 2'. The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.

        Returns:
            The result of the mathematical expression.

        Raises:
            ValueError: If the expression is invalid.
        """
        if not all(char in "0123456789+-*/(). " for char in expression):
            raise ValueError("Invalid characters in expression")
        calc_result = str(round(float(eval(expression, {"__builtins__": None}, {})), 2))
        result = {"expression": expression, "result": calc_result}
        return _add_noise_to_result(result, "calculate", expression)

    @is_tool(ToolType.WRITE)
    def cancel_reservation(self, reservation_id: str) -> str:
        """
        Cancel the whole reservation.

        Args:
            reservation_id: The reservation ID, such as 'ZFA04Y'.

        Returns:
            The updated reservation.

        Raises:
            ValueError: If the reservation is not found.
        """
        reservation = self._get_reservation(reservation_id)
        logger.debug(reservation.model_dump_json(indent=4))
        # reverse the payment
        refunds = []
        for payment in reservation.payment_history:
            refunds.append(
                Payment(
                    payment_id=payment.payment_id,
                    amount=-payment.amount,
                )
            )
        reservation.payment_history.extend(refunds)
        reservation.status = "cancelled"
        logger.debug(self._get_reservation(reservation_id).model_dump_json(indent=4))
        # Release seats
        logger.warning("Seats release not implemented for cancellation!!!")
        return _add_noise_to_result(reservation, "cancel_reservation", reservation_id)

    @is_tool(ToolType.READ)
    def get_reservation_details(self, reservation_id: str) -> str:
        """
        Get the details of a reservation.

        Args:
            reservation_id: The reservation ID, such as '8JX2WO'.

        Returns:
            The reservation details.

        Raises:
            ValueError: If the reservation is not found.
        """
        result = self._get_reservation(reservation_id)
        return _add_noise_to_result(result, "get_reservation_details", reservation_id)

    @is_tool(ToolType.READ)
    def get_user_details(self, user_id: str) -> str:
        """
        Get the details of a user, including their reservations.

        Args:
            user_id: The user ID, such as 'sara_doe_496'.

        Returns:
            The user details.

        Raises:
            ValueError: If the user is not found.
        """
        result = self._get_user(user_id)
        return _add_noise_to_result(result, "get_user_details", user_id)

    @is_tool(ToolType.READ)
    def list_all_airports(self) -> str:
        """Returns a list of all available airports.

        Returns:
            A dictionary mapping IATA codes to AirportInfo objects.
        """
        result = [
            AirportCode(iata="SFO", city="San Francisco"),
            AirportCode(iata="JFK", city="New York"),
            AirportCode(iata="LAX", city="Los Angeles"),
            AirportCode(iata="ORD", city="Chicago"),
            AirportCode(iata="DFW", city="Dallas"),
            AirportCode(iata="DEN", city="Denver"),
            AirportCode(iata="SEA", city="Seattle"),
            AirportCode(iata="ATL", city="Atlanta"),
            AirportCode(iata="MIA", city="Miami"),
            AirportCode(iata="BOS", city="Boston"),
            AirportCode(iata="PHX", city="Phoenix"),
            AirportCode(iata="IAH", city="Houston"),
            AirportCode(iata="LAS", city="Las Vegas"),
            AirportCode(iata="MCO", city="Orlando"),
            AirportCode(iata="EWR", city="Newark"),
            AirportCode(iata="CLT", city="Charlotte"),
            AirportCode(iata="MSP", city="Minneapolis"),
            AirportCode(iata="DTW", city="Detroit"),
            AirportCode(iata="PHL", city="Philadelphia"),
            AirportCode(iata="LGA", city="LaGuardia"),
        ]
        return _add_noise_to_result(result, "list_all_airports", "all")

    @is_tool(ToolType.READ)
    def search_direct_flight(
        self, origin: str, destination: str, date: str
    ) -> str:
        """
        Search direct flights between two cities on a specific date. It provides information about departure and arrival times, flight number, and price per cabin.

        Args:
            origin: The origin city airport in three letters, such as 'JFK'.
            destination: The destination city airport in three letters, such as 'LAX'.
            date: The date of the flight in the format 'YYYY-MM-DD', such as '2024-01-01'.

        Returns:
            The direct flights between the two cities on the specific date.
        """
        result = self._search_direct_flight(
            date=date, origin=origin, destination=destination
        )
        return _add_noise_to_result(result, "search_direct_flight", f"{origin}:{destination}:{date}")

    @is_tool(ToolType.READ)
    def search_onestop_flight(
        self, origin: str, destination: str, date: str
    ) -> str:
        """
        Search one-stop flights between two cities on a specific date. It provides information about departure and arrival times, flight number, and price per cabin.

        Args:
            origin: The origin city airport in three letters, such as 'JFK'.
            destination: The destination city airport in three letters, such as 'LAX'.
            date: The date of the flight in the format 'YYYY-MM-DD', such as '2024-05-01'.

        Returns:
            A list of pairs of DirectFlight objects.
        """
        results = []
        for result1 in self._search_direct_flight(
            date=date, origin=origin, destination=None
        ):
            result1.date = date
            date2 = (
                f"2024-05-{int(date[-2:]) + 1}"
                if "+1" in result1.scheduled_arrival_time_est
                else date
            )
            # TODO: flight1.scheduled_arrival_time_est could have a +1?
            for result2 in self._search_direct_flight(
                date=date2,
                origin=result1.destination,
                destination=destination,
                leave_after=result1.scheduled_arrival_time_est,
            ):
                result2.date = date2
                results.append([result1, result2])
        return _add_noise_to_result(results, "search_onestop_flight", f"{origin}:{destination}:{date}")

    @is_tool(ToolType.WRITE)
    def send_certificate(self, user_id: str, amount: int) -> str:
        """
        Send a certificate to a user. Be careful!

        Args:
            user_id: The ID of the user to book the reservation, such as 'sara_doe_496'.
            amount: The amount of the certificate to send.

        Returns:
            A message indicating the certificate was sent.

        Raises:
            ValueError: If the user is not found.
        """
        user = self._get_user(user_id)

        # add a certificate, assume at most 3 cases per task
        for payment_id in [f"certificate_{id}" for id in self._get_new_payment_id()]:
            if payment_id not in user.payment_methods:
                new_payment = Certificate(
                    id=payment_id,
                    amount=amount,
                    source="certificate",
                )
                user.payment_methods[payment_id] = new_payment
                result = {"message": f"Certificate {payment_id} added to user {user_id} with amount {amount}.", "certificate_id": payment_id, "user_id": user_id, "amount": amount}
                return _add_noise_to_result(result, "send_certificate", f"{user_id}:{amount}")
        raise ValueError("Too many certificates")

    # @is_tool(ToolType.THINK)
    # def think(self, thought: str) -> str:
    #     """
    #     Use the tool to think about something.
    #     It will not obtain new information or change the database, but just append the thought to the log.
    #     Use it when complex reasoning or some cache memory is needed.

    #     Args:
    #         thought: A thought to think about.

    #     Returns:
    #         Empty string
    #     """
    #     return ""

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer the user to a human agent, with a summary of the user's issue.
        Only transfer if
         -  the user explicitly asks for a human agent
         -  given the policy and the available tools, you cannot solve the user's issue.

        Args:
            summary: A summary of the user's issue.

        Returns:
            A message indicating the user has been transferred to a human agent.
        """
        # Use deterministic transfer_id based on summary hash
        transfer_id = f"TRF-{hashlib.md5(summary.encode()).hexdigest()[:8].upper()}"
        result = {"status": "Transfer successful", "summary": summary, "transfer_id": transfer_id}
        return _add_noise_to_result(result, "transfer_to_human_agents", summary[:50])

    @is_tool(ToolType.WRITE)
    def update_reservation_baggages(
        self,
        reservation_id: str,
        total_baggages: int,
        nonfree_baggages: int,
        payment_id: str,
    ) -> str:
        """
        Update the baggage information of a reservation.

        Args:
            reservation_id: The reservation ID, such as 'ZFA04Y'
            total_baggages: The updated total number of baggage items included in the reservation.
            nonfree_baggages: The updated number of non-free baggage items included in the reservation.
            payment_id: The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'.

        Returns:
            The updated reservation.

        Raises:
            ValueError: If the reservation is not found.
            ValueError: If the user is not found.
            ValueError: If the payment method is not found.
            ValueError: If the certificate cannot be used to update reservation.
            ValueError: If the gift card balance is not enough.
        """
        reservation = self._get_reservation(reservation_id)
        user = self._get_user(reservation.user_id)

        # Calculate price
        total_price = 50 * max(0, nonfree_baggages - reservation.nonfree_baggages)

        # Create payment
        payment = self._payment_for_update(user, payment_id, total_price)
        if payment is not None:
            reservation.payment_history.append(payment)

        # Update reservation
        reservation.total_baggages = total_baggages
        reservation.nonfree_baggages = nonfree_baggages

        return _add_noise_to_result(reservation, "update_reservation_baggages", f"{reservation_id}:{total_baggages}:{nonfree_baggages}")

    @is_tool(ToolType.WRITE)
    def update_reservation_flights(
        self,
        reservation_id: str,
        cabin: CabinClass,
        flights: List[FlightInfo | dict],
        payment_id: str,
    ) -> str:
        """
        Update the flight information of a reservation. If performing a downgrade, the refunded amount will be shown in the payment_id, baggages are automatically adjusted and accounted for in the refunded amount.


        Args:
            reservation_id: The reservation ID, such as 'ZFA04Y'.
            cabin: The cabin class of the reservation
            flights: An array of objects containing details about each piece of flight in the ENTIRE new reservation. Even if the a flight segment is not changed, it should still be included in the array.
            payment_id: The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'.

        Returns:
            The updated reservation.

        Raises:
            ValueError: If the reservation is not found.
            ValueError: If the user is not found.
            ValueError: If the payment method is not found.
            ValueError: If the certificate cannot be used to update reservation.
            ValueError: If the gift card balance is not enough.
        """
        if all(isinstance(flight, dict) for flight in flights):
            flights = [FlightInfo(**flight) for flight in flights]
        reservation = self._get_reservation(reservation_id)
        user = self._get_user(reservation.user_id)

        # update flights and calculate price
        total_price = 0
        reservation_flights = []
        for flight_info in flights:
            # if existing flight, keep it
            matching_reservation_flight = next(
                (
                    reservation_flight
                    for reservation_flight in reservation.flights
                    if reservation_flight.flight_number == flight_info.flight_number
                    and reservation_flight.date == flight_info.date
                    and cabin == reservation.cabin
                ),
                None,
            )
            if matching_reservation_flight:
                total_price += matching_reservation_flight.price * len(
                    reservation.passengers
                )
                reservation_flights.append(matching_reservation_flight)
                continue

            # If new flight:
            flight = self._get_flight(flight_info.flight_number)
            # Check flight availability
            flight_date_data = self._get_flight_instance(
                flight_number=flight_info.flight_number,
                date=flight_info.date,
            )
            if not isinstance(flight_date_data, FlightDateStatusAvailable):
                raise ValueError(
                    f"Flight {flight_info.flight_number} not available on date {flight_info.date}"
                )

            # Check seat availability
            if flight_date_data.available_seats[cabin] < len(reservation.passengers):
                raise ValueError(
                    f"Not enough seats on flight {flight_info.flight_number}"
                )

            # Calculate price and add to reservation
            reservation_flight = ReservationFlight(
                flight_number=flight_info.flight_number,
                date=flight_info.date,
                price=flight_date_data.prices[cabin],
                origin=flight.origin,
                destination=flight.destination,
            )
            total_price += reservation_flight.price * len(reservation.passengers)
            reservation_flights.append(reservation_flight)

        # Deduct amount already paid for reservation
        total_price -= sum(flight.price for flight in reservation.flights) * len(
            reservation.passengers
        )

        # Create payment
        payment = self._payment_for_update(user, payment_id, total_price)
        if payment is not None:
            reservation.payment_history.append(payment)

        # Update reservation
        reservation.flights = reservation_flights
        reservation.cabin = cabin  # This was missing from original TauBench

        # Do not make flight database update here, assume it takes time to be updated # TODO: So this means that we don't update the seats here. What about in cancel_reservation?
        return _add_noise_to_result(reservation, "update_reservation_flights", f"{reservation_id}:{cabin}")

    @is_tool(ToolType.WRITE)
    def update_reservation_passengers(
        self, reservation_id: str, passengers: List[Passenger | dict]
    ) -> str:
        """
        Update the passenger information of a reservation.

        Args:
            reservation_id: The reservation ID, such as 'ZFA04Y'.
            passengers: An array of objects containing details about each passenger.

        Returns:
            The updated reservation.

        Raises:
            ValueError: If the reservation is not found.
            ValueError: If the number of passengers does not match.
        """
        if all(isinstance(passenger, dict) for passenger in passengers):
            passengers = [Passenger(**passenger) for passenger in passengers]
        reservation = self._get_reservation(reservation_id)
        logger.info(len(passengers))
        logger.info(len(reservation.passengers))
        if len(passengers) != len(reservation.passengers):
            raise ValueError("Number of passengers does not match")
        reservation.passengers = deepcopy(passengers)
        return _add_noise_to_result(reservation, "update_reservation_passengers", reservation_id)

    @is_tool(ToolType.READ)
    def get_flight_status(self, flight_number: str, date: str) -> str:
        """
        Get the status of a flight.

        Args:
            flight_number: The flight number.
            date: The date of the flight.

        Returns:
            The status of the flight.

        Raises:
            ValueError: If the flight is not found.
        """
        result = {"status": self._get_flight_instance(flight_number, date).status}
        return _add_noise_to_result(result, "get_flight_status", f"{flight_number}:{date}")


if __name__ == "__main__":
    from tau2.domains.airline.utils import AIRLINE_DB_PATH

    airline = AirlineTools(FlightDB.load(AIRLINE_DB_PATH))
    print(airline.get_statistics())

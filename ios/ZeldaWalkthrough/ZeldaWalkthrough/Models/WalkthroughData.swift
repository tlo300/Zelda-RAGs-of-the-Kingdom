// WalkthroughData.swift
// Static chapter and section structure for the TotK walkthrough. Content is blank pending population.

import Foundation

enum ChapterCategory: String, CaseIterable {
    case mainQuest = "Main Quest"
    case sideContent = "Side Content"
    case reference = "Reference"
}

struct WalkthroughSection: Identifiable, Hashable {
    let id: String
    let title: String
    var content: String = ""
}

struct WalkthroughChapter: Identifiable {
    let id: String
    let title: String
    let category: ChapterCategory
    let icon: String
    let sections: [WalkthroughSection]
}

enum WalkthroughData {
    static let chapters: [WalkthroughChapter] = [

        // MARK: - Main Quest

        WalkthroughChapter(
            id: "getting-started",
            title: "Getting Started",
            category: .mainQuest,
            icon: "star.circle",
            sections: [
                WalkthroughSection(id: "gs-controls", title: "Controls & UI Overview"),
                WalkthroughSection(id: "gs-abilities", title: "Link's Core Abilities"),
                WalkthroughSection(id: "gs-tips", title: "Early Game Tips"),
            ]
        ),

        WalkthroughChapter(
            id: "great-sky-island",
            title: "The Great Sky Island",
            category: .mainQuest,
            icon: "cloud",
            sections: [
                WalkthroughSection(id: "gsi-wake", title: "Waking Up"),
                WalkthroughSection(id: "gsi-ukouh", title: "Ukouh Shrine — Ultrahand"),
                WalkthroughSection(id: "gsi-inisa", title: "In-isa Shrine — Fuse"),
                WalkthroughSection(id: "gsi-gutanbac", title: "Gutanbac Shrine — Ascend"),
                WalkthroughSection(id: "gsi-nachoyah", title: "Nachoyah Shrine — Recall"),
                WalkthroughSection(id: "gsi-temple", title: "Reach the Temple of Time"),
                WalkthroughSection(id: "gsi-descent", title: "Descend to Hyrule"),
            ]
        ),

        WalkthroughChapter(
            id: "arriving-in-hyrule",
            title: "Arriving in Hyrule",
            category: .mainQuest,
            icon: "mappin.circle",
            sections: [
                WalkthroughSection(id: "aih-lookout", title: "Lookout Landing"),
                WalkthroughSection(id: "aih-shelter", title: "Emergency Shelter"),
                WalkthroughSection(id: "aih-camera", title: "Camera Work in the Depths"),
                WalkthroughSection(id: "aih-castle", title: "Infiltrating Hyrule Castle"),
                WalkthroughSection(id: "aih-phenomena", title: "The Regional Phenomena"),
            ]
        ),

        WalkthroughChapter(
            id: "wind-temple",
            title: "Rito Village & the Wind Temple",
            category: .mainQuest,
            icon: "wind",
            sections: [
                WalkthroughSection(id: "wt-rito", title: "Reach Rito Village"),
                WalkthroughSection(id: "wt-tulin", title: "Tulin of Rito Village"),
                WalkthroughSection(id: "wt-storm", title: "Approach the Storm"),
                WalkthroughSection(id: "wt-temple", title: "Wind Temple — Puzzles"),
                WalkthroughSection(id: "wt-boss", title: "Boss: Colgera"),
            ]
        ),

        WalkthroughChapter(
            id: "water-temple",
            title: "Zora's Domain & the Water Temple",
            category: .mainQuest,
            icon: "drop",
            sections: [
                WalkthroughSection(id: "wat-zora", title: "Reach Zora's Domain"),
                WalkthroughSection(id: "wat-sidon", title: "Sidon of the Zora"),
                WalkthroughSection(id: "wat-sludge", title: "Clear the Sludge"),
                WalkthroughSection(id: "wat-temple", title: "Water Temple — Puzzles"),
                WalkthroughSection(id: "wat-boss", title: "Boss: Mucktorok"),
            ]
        ),

        WalkthroughChapter(
            id: "fire-temple",
            title: "Goron City & the Fire Temple",
            category: .mainQuest,
            icon: "flame",
            sections: [
                WalkthroughSection(id: "ft-goron", title: "Reach Goron City"),
                WalkthroughSection(id: "ft-yunobo", title: "Yunobo of Goron City"),
                WalkthroughSection(id: "ft-mine", title: "YunoboCo HQ & the Mine"),
                WalkthroughSection(id: "ft-temple", title: "Fire Temple — Puzzles"),
                WalkthroughSection(id: "ft-boss", title: "Boss: Marbled Gohma"),
            ]
        ),

        WalkthroughChapter(
            id: "lightning-temple",
            title: "Gerudo Town & the Lightning Temple",
            category: .mainQuest,
            icon: "bolt",
            sections: [
                WalkthroughSection(id: "lt-gerudo", title: "Reach Gerudo Town"),
                WalkthroughSection(id: "lt-riju", title: "Riju of Gerudo Town"),
                WalkthroughSection(id: "lt-yiga", title: "Infiltrate the Yiga Clan Hideout"),
                WalkthroughSection(id: "lt-temple", title: "Lightning Temple — Puzzles"),
                WalkthroughSection(id: "lt-boss", title: "Boss: Queen Gibdo"),
            ]
        ),

        WalkthroughChapter(
            id: "master-sword",
            title: "The Master Sword",
            category: .mainQuest,
            icon: "sparkles",
            sections: [
                WalkthroughSection(id: "ms-tears", title: "The Dragon's Tears — All Memories"),
                WalkthroughSection(id: "ms-rings", title: "Secret of the Ring Ruins"),
                WalkthroughSection(id: "ms-dragon", title: "Reach the Light Dragon"),
                WalkthroughSection(id: "ms-obtain", title: "Obtain the Master Sword"),
            ]
        ),

        WalkthroughChapter(
            id: "hyrule-castle",
            title: "Hyrule Castle",
            category: .mainQuest,
            icon: "building.columns",
            sections: [
                WalkthroughSection(id: "hc-crisis", title: "Crisis at Hyrule Castle"),
                WalkthroughSection(id: "hc-guidance", title: "Guidance from Ages Past"),
                WalkthroughSection(id: "hc-ascent", title: "Ascend the Castle"),
            ]
        ),

        WalkthroughChapter(
            id: "final-battle",
            title: "The Final Battle",
            category: .mainQuest,
            icon: "crown",
            sections: [
                WalkthroughSection(id: "fb-gloom", title: "Gloom's Origin"),
                WalkthroughSection(id: "fb-army", title: "The Demon King's Army"),
                WalkthroughSection(id: "fb-king", title: "Boss: The Demon King"),
                WalkthroughSection(id: "fb-dragons", title: "The Dragon Showdown"),
            ]
        ),

        // MARK: - Side Content

        WalkthroughChapter(
            id: "shrines",
            title: "Shrines",
            category: .sideContent,
            icon: "diamond",
            sections: [
                WalkthroughSection(id: "shr-sky", title: "Sky Islands Shrines"),
                WalkthroughSection(id: "shr-eldin", title: "Eldin & Akkala Shrines"),
                WalkthroughSection(id: "shr-lanayru", title: "Lanayru & Necluda Shrines"),
                WalkthroughSection(id: "shr-faron", title: "Faron Shrines"),
                WalkthroughSection(id: "shr-central", title: "Central Hyrule Shrines"),
                WalkthroughSection(id: "shr-gerudo", title: "Gerudo & Rito Shrines"),
                WalkthroughSection(id: "shr-depths", title: "Depths Shrines"),
            ]
        ),

        WalkthroughChapter(
            id: "side-quests",
            title: "Side Quests",
            category: .sideContent,
            icon: "list.bullet.clipboard",
            sections: [
                WalkthroughSection(id: "sq-stables", title: "Stable Quests"),
                WalkthroughSection(id: "sq-addison", title: "Addison's Hudson Signs"),
                WalkthroughSection(id: "sq-zelda", title: "Potential Princess Sightings"),
                WalkthroughSection(id: "sq-lurelin", title: "Lurelin Village Restoration"),
                WalkthroughSection(id: "sq-phantom", title: "The Phantom Ganon Armor"),
            ]
        ),

        WalkthroughChapter(
            id: "side-adventures",
            title: "Side Adventures",
            category: .sideContent,
            icon: "figure.walk",
            sections: [
                WalkthroughSection(id: "sa-sages", title: "Sage's Will Locations"),
                WalkthroughSection(id: "sa-dispelling", title: "Dispelling Gloom"),
                WalkthroughSection(id: "sa-construct", title: "Steward Construct Quests"),
            ]
        ),

        WalkthroughChapter(
            id: "collectibles",
            title: "Collectibles & Completion",
            category: .sideContent,
            icon: "star.circle.fill",
            sections: [
                WalkthroughSection(id: "col-korok", title: "Korok Seeds (1000)"),
                WalkthroughSection(id: "col-bubbul", title: "Bubbulfrogs"),
                WalkthroughSection(id: "col-bargainer", title: "Bargainer Statues"),
                WalkthroughSection(id: "col-bosses", title: "Hinoxes, Taluses & Moldugas"),
                WalkthroughSection(id: "col-dragons", title: "Dragon Parts Farming"),
            ]
        ),

        // MARK: - Reference

        WalkthroughChapter(
            id: "tips-and-tricks",
            title: "Tips & Tricks",
            category: .reference,
            icon: "lightbulb",
            sections: [
                WalkthroughSection(id: "tt-combat", title: "Combat Tips"),
                WalkthroughSection(id: "tt-building", title: "Ultrahand & Building"),
                WalkthroughSection(id: "tt-rupees", title: "Farming Rupees"),
                WalkthroughSection(id: "tt-cooking", title: "Cooking & Elixirs"),
                WalkthroughSection(id: "tt-armor", title: "Best Armor Sets"),
            ]
        ),

        WalkthroughChapter(
            id: "equipment",
            title: "Weapons & Equipment",
            category: .reference,
            icon: "shield",
            sections: [
                WalkthroughSection(id: "eq-weapons", title: "Best Weapon Fusions"),
                WalkthroughSection(id: "eq-shields", title: "Shield Surfing & Fusions"),
                WalkthroughSection(id: "eq-bows", title: "Bow & Arrow Types"),
                WalkthroughSection(id: "eq-special", title: "Unique & Unbreakable Weapons"),
            ]
        ),
    ]

    static func chapters(for category: ChapterCategory) -> [WalkthroughChapter] {
        chapters.filter { $0.category == category }
    }
}

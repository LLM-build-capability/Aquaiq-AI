# Exercise 1 – Library Book Borrowing
 
| | |
|---|---|
| **Author:** |Prem Kumar Reddy Kothakapu |
|**Date:** | 11-May-2026 |
|**Exercise:** |Architecture Modelling with ArchiMate |
|**System:** |Borrowing a physical book from a public library using a library card.|
## System overview
 
The system supports a **Patron** (library user) and a **Librarian** to complete a book borrowing transactions. The core business service is **Borrow a Book**. The business process **Borrow Book Process** covers: searching for a book, checking its avaliability, borrowing it, and optionally paying a fine. The library uses a card-based identification system (library card) to authorise each transaction.
## Business layer
 
The Patron and Librarian are both **assigned** to the Borrow a Book service. There is also a **Library Membership** object, and I used a **composition** relationship from Patron to Membership – just to meet the assignment requirement (it feels a little forced, but it works). The Borrow Book Process **realises** the Borrow a Book service. The process **accesses** two business objects: **Book Record** and **Patron Record**.
 
## Application layer
 
The **Library Catalogue System** (application component) provides a **Search and Borrow Service**. It also contains a **Fine Payment Service** (composition). The catalogue system **serves** the Search and Borrow Service. The system **accesses** three data objects: **Book Database**, **Patron Database**, and **Fine Transaction**. There is also an **External Book Database** (external application service) that **serves** the catalogue system.
 
## Technology layer
 
The catalogue system runs on a **Library Server Node** (Node). The **Library Card Reader** (Device) is attached to the same node. I used **flow** relationships from the server node to the card reader and to the **Library Network** (technology service). The network then **serves** the external book database.
 
## Relationships used
 
- **Assignment** (Patron → Borrow a Book, Librarian → Borrow a Book)
- **Composition** (Patron → Library Membership)
- **Realization** (Borrow Book Process → Borrow a Book)
- **Serving** (Catalogue System → Search and Borrow Service; External DB → Catalogue System)
- **Access** (Process → Book Record, Patron Record, etc.)
- **Flow** (Server Node → Card Reader, Server Node → Network)

I have included all the required relationship types. The model shows the whole flow from the user’s request down to the physical hardware and external network.